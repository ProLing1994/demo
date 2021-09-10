import librosa
import numpy as np
import pcen
import random
import sys
import struct
from scipy.ndimage.morphology import binary_dilation
from typing import Optional, Union
import webrtcvad
import torch

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/SV')
from config.hparams import *

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_feature_pyimpl import Feature


def preprocess_wav(fpath_or_wav: Union[str, np.ndarray],
                   sampling_rate: Union[int],
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = True):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav if needed
    if source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    if normalize:
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    if trim_silence:
        wav = trim_long_silences(wav, sampling_rate)
    
    return wav


def trim_long_silences(wav, sampling_rate):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))


def dataset_alignment(cfg, data, bool_replicate=False):
    # init
    desired_samples = int(cfg.dataset.sample_rate * cfg.dataset.clip_duration_ms / 1000)

    # alignment data
    if len(data) < desired_samples:
        data_length = len(data)
        if bool_replicate:
            tile_size = (desired_samples // data_length) + 1
            data = np.tile(data, tile_size)[:desired_samples]
        else:
            data_offset = np.random.randint(0, desired_samples - data_length)
            data = np.pad(data, (data_offset, 0), "constant")
            data = np.pad(data, (0, desired_samples - data_length - data_offset), "constant")

    if len(data) > desired_samples:
        data_offset = np.random.randint(0, len(data) - desired_samples)
        data = data[data_offset:(data_offset + desired_samples)]

    assert len(data) == desired_samples, "[ERROR:] Something wronge with audio length, please check"
    return data


def dataset_augmentation_volume_speed(cfg, data):
    # init
    speed_list = cfg.dataset.augmentation.speed
    volume_list = cfg.dataset.augmentation.volume

    speed = np.random.uniform(speed_list[0], speed_list[1])  
    volume = np.random.uniform(volume_list[0], volume_list[1])  
    
    # speed > 1, 加快速度
    # speed < 1, 放慢速度
    data = librosa.effects.time_stretch(data, speed)

    # 音量大小调节
    data = data * volume
    return data

def dataset_augmentation_pitch(cfg, data):
    # init
    pitch_list = cfg.dataset.augmentation.pitch 

    pitch = np.random.randint(pitch_list[0], pitch_list[1])  

    # 音调调节
    data = librosa.effects.pitch_shift(data, sr=cfg.dataset.sample_rate, n_steps=pitch)
    return data

def dataset_add_synthetic_noise(cfg, data):
    # init
    synthetic_type = cfg.dataset.augmentation.synthetic_type
    synthetic_frequency = cfg.dataset.augmentation.synthetic_frequency
    synthetic_scale = cfg.dataset.augmentation.synthetic_scale
    synthetic_prob = cfg.dataset.augmentation.synthetic_prob

    if not synthetic_frequency > 0:
        return data

    if np.random.uniform(0, 1) < synthetic_frequency:
        if synthetic_type == "white":
            scale = synthetic_scale * np.random.uniform(0, 1)
            synthetic_noise = np.random.normal(loc=0, scale=scale, size=data.shape)
            data = (data + synthetic_noise)
        elif synthetic_type == "salt_pepper": 
            prob = synthetic_prob * np.random.uniform(0, 1)
            synthetic_noise = np.random.binomial(1, prob, size=data.shape) - np.random.binomial(1, prob, size=data.shape) 
            data = (data + synthetic_noise)

    # data clip
    data = np.clip(data, -1.0, 1.0)
    return data

def dataset_add_noise(cfg, data, background_data):          
    # init
    desired_samples = int(cfg.dataset.sample_rate * cfg.dataset.clip_duration_ms / 1000)
    background_frequency = cfg.dataset.augmentation.background_frequency 
    background_volume = cfg.dataset.augmentation.background_volume

    if not background_frequency > 0:
        return data
    
    assert len(background_data) > 0, "[ERROR:] Something wronge with background data, please check"

    # init
    background_clipped = np.zeros(desired_samples)
    background_volume = 0

    background_idx = np.random.randint(len(background_data))
    background_sample = background_data[background_idx]

    # alignment background data
    background_clipped = dataset_alignment(cfg, background_sample, True) 

    if np.random.uniform(0, 1) < background_frequency:
        background_volume = np.random.uniform(0, background_volume)

    data_max_value = data.max()
    background_max_value = (background_volume * background_clipped).max() * 0.8
    if background_max_value < data_max_value:
        data = background_volume * background_clipped + data

    # add synthetic noise
    data = dataset_add_synthetic_noise(cfg, data)

    # data clip
    data = np.clip(data, -1.0, 1.0)
    return data

def dataset_augmentation_waveform(cfg, data, background_data):
    # data augmentation
    if cfg.dataset.augmentation.speed_volume_on:
        data = dataset_augmentation_volume_speed(cfg, data)
    
    # data augmentation
    if cfg.dataset.augmentation.pitch_on:
        data = dataset_augmentation_pitch(cfg, data)

    # alignment data
    data = dataset_alignment(cfg, data) 

    # add time_shift
    time_shift_ms = cfg.dataset.augmentation.time_shift_ms
    time_shift_samples = int(cfg.dataset.sample_rate * time_shift_ms / 1000)

    if time_shift_samples > 0:
        time_shift_amount = np.random.randint(-time_shift_samples, time_shift_samples)
        time_shift_left = -min(0, time_shift_amount)
        time_shift_right = max(0, time_shift_amount)
        data = np.pad(data, (time_shift_left, time_shift_right), "constant")
        data = data[:len(data) - time_shift_left] if time_shift_left else data[time_shift_right:]

    # add noise
    data = dataset_add_noise(cfg, data, background_data)
    return data

def add_frequence_mask(spec, F=27, num_masks=1, replace_with_zero=False, static=False):
    num_mel_channels = spec.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)

        if static:
            f = F

        f_start = random.randrange(0, num_mel_channels - f)
        f_end = f_start + f

        # avoids randrange error if values are equal and range is empty
        if (f_start == f_start + f):
            return spec

        if (replace_with_zero):
            spec[:, f_start:f_end] = 0
        else:
            spec[:, f_start:f_end] = spec.mean()

    return spec


def add_time_mask(spec, T=40, num_masks=1, replace_with_zero=False, static=False):
    len_spectro = spec.shape[0]
    
    for i in range(0, num_masks):
        t = random.randrange(0, T)

        if static:
            t = T

        t_start = random.randrange(0, len_spectro - t)
        t_end = t_start + t

        # avoids randrange error if values are equal and range is empty
        if (t_start == t_start + t):
            return spec

        if (replace_with_zero): 
            spec[t_start:t_end] = 0
        else: 
            spec[t_start:t_end] = spec.mean()

    return spec


def dataset_augmentation_spectrum(cfg, audio_data):
    # init
    F = cfg.dataset.augmentation.F
    T = cfg.dataset.augmentation.T
    num_masks = cfg.dataset.augmentation.num_masks

    # add SpecAugment
    audio_data = add_frequence_mask(audio_data, F=F, num_masks=num_masks, replace_with_zero=True)
    audio_data = add_time_mask(audio_data, T=T, num_masks=num_masks, replace_with_zero=True)
    return audio_data


def audio_preprocess(cfg, data):
    # init 
    audio_preprocess_type = cfg.dataset.preprocess
    audio_processor = AudioPreprocessor(sr=cfg.dataset.sample_rate,
                                        n_mels=cfg.dataset.feature_bin_count,
                                        nfilt=cfg.dataset.nfilt,
                                        winlen=cfg.dataset.window_size_ms / 1000, 
                                        winstep=cfg.dataset.window_stride_ms / 1000,
                                        data_length=cfg.dataset.clip_duration_ms / 1000)

    # check
    assert audio_preprocess_type in ["pcen", "fbank", "fbank_cpu"], "[ERROR:] Audio preprocess type is wronge, please check"

    # preprocess
    # if self.audio_preprocess_type == "mfcc":
    #     audio_data = self.audio_processor.compute_mfccs(data)
    if audio_preprocess_type == "pcen":
        audio_data = audio_processor.compute_pcen(data)
    elif audio_preprocess_type == "fbank":
        audio_data = audio_processor.compute_fbanks(data)
    elif audio_preprocess_type == "fbank_cpu":
        audio_data = audio_processor.compute_fbanks_cpu(data, cfg.dataset.augmentation.vtlp_on)
    return audio_data


class AudioPreprocessor(object):
    def __init__(self, sr=16000, n_mels=40, nfilt=40, n_dct_filters=40, f_max=4000, f_min=20, winlen=0.032, winstep=0.010, data_length=2):
        super().__init__()
        self.sr = sr
        self.n_mels = n_mels
        self.nfilt = nfilt
        self.n_dct_filters = n_dct_filters
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min        
        self.winlen = winlen
        self.winstep = winstep
        self.win_length = int(self.sr * self.winlen)
        self.hop_length = int(self.sr * self.winstep)
        self.data_length = data_length
        self.pcen_transform = pcen.StreamingPCENTransform(n_mels=self.n_mels, n_fft=self.win_length, hop_length=self.hop_length, trainable=True)

    def compute_fbanks(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.win_length,
            fmin=self.f_min,
            fmax=self.f_max)
        data[data > 0] = np.log(data[data > 0])
        data = data.T
        return data

    def compute_pcen(self, data):
        data = torch.from_numpy(np.expand_dims(data, axis=0))
        data = self.pcen_transform(data)
        self.pcen_transform.reset()
        data = data.detach().numpy()
        data = data.reshape(-1, 40)
        return data

    def compute_fbanks_cpu(self, data, bool_vtlp_augmentation=False):
        # data to numpy
        data = data * pow(2,15)
        data = data.astype(int)
        # print(data[:10])
        
        # compute fbank cpu
        featurefbanks_cpu = Feature(sample_rate=self.sr, data_length=self.data_length, feature_freq=self.n_mels, nfilt=self.nfilt, winlen=self.winlen , winstep=self.winstep)
        featurefbanks_cpu.get_mel_int_feature(data, len(data), bool_vtlp_augmentation)
        feature_data = featurefbanks_cpu.copy_mfsc_feature_int_to()
        return feature_data