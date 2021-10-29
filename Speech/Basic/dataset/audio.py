import librosa
import numpy as np
import pcen
import sys
import struct
from scipy import signal
from scipy.ndimage.morphology import binary_dilation
import soundfile as sf
from typing import Optional, Union
import torch
import webrtcvad

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from ASR.impl.asr_feature_pyimpl import Feature

from Basic.config import hparams


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr): 
    sf.write(path, wav, sr)


def preprocess_wav(fpath_or_wav: Union[str, np.ndarray],
                   sampling_rate: Union[int],
                   source_sr: Optional[int] = None,
                   bool_normalize: Optional[bool] = True,
                   bool_trim_silence: Optional[bool] = True):
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
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    if bool_normalize:
        wav = normalize_volume(wav, hparams.audio_norm_target_dBFS, increase_only=True)
    if bool_trim_silence:
        # wav = trim_silence(wav)
        wav = trim_long_silences(wav, sampling_rate)
    
    return wav


def trim_silence(wav):
    wav = librosa.effects.trim(wav, top_db=hparams.trim_top_db, frame_length=hparams.trim_fft_size, hop_length=hparams.trim_hop_size)[0]
    return wav


def trim_long_silences(wav, sampling_rate):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (hparams.vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * hparams.int16_max)).astype(np.int16))
    
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
    
    audio_mask = moving_average(voice_flags, hparams.vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(hparams.vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    """
    音频音量大小归一化，将音频规整到目标分贝 target_dBFS
    计算公式：$N_{dB}=10 \lg \frac{pi}{p0}$，${pi}$、${p0}$ 为功率的单位
             $N_{dB}=20 \lg \frac{pi}{p0}$，${pi}$、${p0}$ 为幅值的单位
    """
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")

    # 计算公式：$N_{dB}=10 \lg \frac{pi}{p0}$，${pi}$、${p0}$ 为功率的单位
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav

    # 计算公式：$N_{dB}=20 \lg \frac{pi}{p0}$，${pi}$、${p0}$ 为幅值的单位
    return wav * (10 ** (dBFS_change / 20))


class AudioPreprocessor(object):
    def __init__(self, sr=16000, n_mels=40, nfilt=40, n_dct_filters=40, f_max=4000, f_min=20, winlen=0.032, winstep=0.010):
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
        self.pcen_transform = pcen.StreamingPCENTransform(n_mels=self.n_mels, n_fft=self.win_length, hop_length=self.hop_length, trainable=True)

        # init 
        self.mel_basis = None
        self.inv_mel_basis = None

    def compute_fbanks(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels)
        data = data.astype(np.float32).T
        return data

    def compute_fbanks_log(self, data):
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

    def fbank_log_manual(self, data):
        D = self.stft(self.preemphasis(data, hparams.preemphasis, hparams.preemphasize))
        S = self.amp_to_db(self.linear_to_mel(np.abs(D))) - hparams.ref_level_db
        
        if hparams.signal_normalization:
            return self.normalize(S).T
        return S.T

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
        featurefbanks_cpu = Feature(sample_rate=self.sr, feature_freq=self.n_mels, nfilt=self.nfilt, winlen=self.winlen , winstep=self.winstep)
        featurefbanks_cpu.get_mel_int_feature(data, bool_vtlp_augmentation)
        feature_data = featurefbanks_cpu.copy_mfsc_feature_int_to()
        return feature_data

    def preemphasis(self, wav, k, preemphasize=True):
        if preemphasize:
            return signal.lfilter([1, -k], [1], wav)
        return wav

    def inv_preemphasis(self, wav, k, inv_preemphasize=True):
        if inv_preemphasize:
            return signal.lfilter([1], [1, -k], wav)
        return wav

    def stft(self, y):
        return librosa.stft(y=y, n_fft=self.win_length, hop_length=self.hop_length, win_length=self.win_length)

    def istft(self, y):
        return librosa.istft(y, hop_length=self.hop_length, win_length=self.win_length)

    def linear_to_mel(self, spectogram):
        if self.mel_basis is None:
            self.mel_basis = self.build_mel_basis()
        return np.dot(self.mel_basis, spectogram)

    def mel_to_linear(self, mel_spectrogram):
        if self.inv_mel_basis is None:
            self.inv_mel_basis = np.linalg.pinv(self.build_mel_basis())
        return np.maximum(1e-10, np.dot(self.inv_mel_basis, mel_spectrogram))

    def build_mel_basis(self):
        assert self.f_max <= self.sr // 2
        return librosa.filters.mel(self.sr, self.win_length, n_mels=self.n_mels,
                                fmin=self.f_min, fmax=self.f_max)

    def amp_to_db(self, x):
        min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def db_to_amp(self, x):
        return np.power(10.0, (x) * 0.05)

    def normalize(self, S):
        if hparams.allow_clipping_in_normalization:
            if hparams.symmetric_mels:
                return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
                            -hparams.max_abs_value, hparams.max_abs_value)
            else:
                return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)
        
        assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
        if hparams.symmetric_mels:
            return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
        else:
            return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))

    def denormalize(self, D):
        if hparams.allow_clipping_in_normalization:
            if hparams.symmetric_mels:
                return (((np.clip(D, -hparams.max_abs_value, hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
                        + hparams.min_level_db)
            else:
                return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
        
        if hparams.symmetric_mels:
            return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
        else:
            return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

    def griffin_lim(self, S):
        """librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        """
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self.istft(S_complex * angles)
        for i in range(hparams.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self.stft(y)))
            y = self.istft(S_complex * angles)
        return y


def compute_mel_spectrogram(cfg, data):
    # init 
    audio_preprocess_type = cfg.dataset.preprocess
    audio_processor = AudioPreprocessor(sr=cfg.dataset.sample_rate,
                                        n_mels=cfg.dataset.feature_bin_count,
                                        nfilt=cfg.dataset.nfilt,
                                        f_max=cfg.dataset.fmax, 
                                        f_min=cfg.dataset.fmin,
                                        winlen=cfg.dataset.window_size_ms / 1000, 
                                        winstep=cfg.dataset.window_stride_ms / 1000)

    # check
    assert audio_preprocess_type in ["fbank", "fbank_log", "fbank_log_manual", "pcen", "fbank_cpu"], "[ERROR:] Audio preprocess type is wronge, please check"

    # preprocess
    if audio_preprocess_type == "fbank":
        audio_data = audio_processor.compute_fbanks(data)
    elif audio_preprocess_type == "fbank_log":
        audio_data = audio_processor.compute_fbanks_log(data)
    elif audio_preprocess_type == "fbank_log_manual":
        audio_data = audio_processor.fbank_log_manual(data)
    elif audio_preprocess_type == "pcen":
        audio_data = audio_processor.compute_pcen(data)
    elif audio_preprocess_type == "fbank_cpu":
        if cfg.dataset.augmentation.on and cfg.dataset.augmentation.vtlp_on:
            audio_data = audio_processor.compute_fbanks_cpu(data, cfg.dataset.augmentation.vtlp_on)
        else:
            audio_data = audio_processor.compute_fbanks_cpu(data)
    return audio_data.astype(np.float32)


def compute_inv_mel_spectrogram(cfg, mel):
    """Converts mel spectrogram to waveform using librosa"""
    # init 
    audio_preprocess_type = cfg.dataset.preprocess
    assert audio_preprocess_type == "fbank_log_manual", "only support audio preprocess type: fbank_log_manual"

    audio_processor = AudioPreprocessor(sr=cfg.dataset.sample_rate,
                                        n_mels=cfg.dataset.feature_bin_count,
                                        nfilt=cfg.dataset.nfilt,
                                        f_max=cfg.dataset.fmax, 
                                        f_min=cfg.dataset.fmin,
                                        winlen=cfg.dataset.window_size_ms / 1000, 
                                        winstep=cfg.dataset.window_stride_ms / 1000)

    # denormalize
    if hparams.signal_normalization:
        D = audio_processor.denormalize(mel)
    else:
        D = mel
    
    # mel to linear
    S = audio_processor.mel_to_linear(audio_processor.db_to_amp(D + hparams.ref_level_db))  # Convert back to linear
    
    # griffin_lim
    return audio_processor.inv_preemphasis(audio_processor.griffin_lim(S ** hparams.power), hparams.preemphasis, hparams.preemphasize)