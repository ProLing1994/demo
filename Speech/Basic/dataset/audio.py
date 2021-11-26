import librosa
from librosa.filters import window_bandwidth
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


def load_wav(path, sampling_rate):
    return librosa.core.load(path, sampling_rate=sampling_rate)[0]


def save_wav(wav, path, sampling_rate): 
    sf.write(path, wav, sampling_rate)


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
        wav, source_sr = librosa.load(str(fpath_or_wav), sampling_rate=None)
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


def trim_silence_old(wav):
    # Whether to trim the start and end of silence.
    wav = librosa.effects.trim(wav, top_db=hparams.trim_top_db, frame_length=hparams.trim_fft_size, hop_length=hparams.trim_hop_size)[0]
    return wav


def trim_silence(wav, trim_threshold_in_db, trim_frame_size, trim_hop_size):
    # Whether to trim the start and end of silence.
    wav, _ = librosa.effects.trim(wav, top_db=trim_threshold_in_db, frame_length=trim_frame_size, hop_length=trim_hop_size)
    return wav


def trim_long_silences(wav, sampling_rate, return_mask_bool=False):
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
    
    if return_mask_bool:
        return wav[audio_mask == True], audio_mask
    else:
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


class ComputeMel(object):
    def __init__(self, 
                sampling_rate=16000, 
                fft_size=1024, hop_size=256, win_length=1024, window="hann",
                num_mels=80, num_filts=80, num_dct_filters=80, 
                fmax=4000, fmin=20,
                eps=1e-10, log_base=10.0):
        super().__init__()
        self.sampling_rate = sampling_rate

        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = window

        self.num_mels = num_mels
        self.num_filts = num_filts
        self.num_dct_filters = num_dct_filters

        self.fmax = sampling_rate // 2 if fmax is None else fmax
        self.fmin = 0 if fmin is None else fmin

        self.eps = eps
        self.log_base = log_base
  
        self.pcen_transform = pcen.StreamingPCENTransform(n_mels=self.num_mels, n_fft=self.win_length, hop_size=self.hop_size, trainable=True)

        # init 
        self.mel_basis = None
        self.inv_mel_basis = None

    def compute_fbanks(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sampling_rate=self.sampling_rate,
            n_fft=self.win_length,
            hop_size=self.hop_size,
            num_mels=self.num_mels)
        data = data.astype(np.float32).T
        return data

    def compute_fbanks_log(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sampling_rate=self.sampling_rate,
            num_mels=self.num_mels,
            hop_size=self.hop_size,
            n_fft=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax)
        data[data > 0] = np.log(data[data > 0])
        data = data.T
        return data

    def fbank_nopreemphasis_log_manual(self, data):
        D = self.stft(data)
        S = self.amp_to_db(self.linear_to_mel(np.abs(D), min_level=self.eps))
        return S.T

    def compute_fbanks_preemphasis_log_manual(self, data):
        D = self.stft(self.preemphasis(data, hparams.preemphasis, hparams.preemphasize))

        min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
        S = self.amp_to_db(self.linear_to_mel(np.abs(D), min_level=min_level), multi_coef=20) - hparams.ref_level_db
        
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
        featurefbanks_cpu = Feature(sample_rate=self.sampling_rate, feature_freq=self.num_mels, num_filts=self.num_filts, winlen=self.winlen , winstep=self.winstep)
        featurefbanks_cpu.get_mel_int_feature(data, bool_vtlp_augmentation)
        feature_data = featurefbanks_cpu.copy_mfsc_feature_int_to()
        return feature_data

    @classmethod
    def preemphasis(self, wav, k, preemphasize=True):
        if preemphasize:
            return signal.lfilter([1, -k], [1], wav)
        return wav

    @classmethod
    def inv_preemphasis(self, wav, k, inv_preemphasize=True):
        if inv_preemphasize:
            return signal.lfilter([1], [1, -k], wav)
        return wav

    def stft(self, y):
        return librosa.stft(y=y, n_fft=self.fft_size, hop_length=self.hop_size, win_length=self.win_length, window=self.window, pad_mode='reflect')

    def istft(self, y):
        return librosa.istft(y, hop_length=self.hop_size, win_length=self.win_length)

    def linear_to_mel(self, spectogram, min_level):
        if self.mel_basis is None:
            self.mel_basis = self.build_mel_basis()
        return np.maximum(min_level, np.dot(self.mel_basis, spectogram))

    def mel_to_linear(self, mel_spectrogram):
        if self.inv_mel_basis is None:
            self.inv_mel_basis = np.linalg.pinv(self.build_mel_basis())
        return np.maximum(self.eps, np.dot(self.inv_mel_basis, mel_spectrogram))

    def build_mel_basis(self):
        assert self.fmax <= self.sampling_rate // 2
        return librosa.filters.mel(self.sampling_rate, n_fft=self.fft_size, n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax)

    def amp_to_db(self, x, log_base=10.0, multi_coef=1.0):
        if log_base is None:
            return multi_coef * np.log(x)
        elif log_base == 10.0:
            return multi_coef * np.log10(x)
        elif log_base == 2.0:
            return multi_coef * np.log2(x)
        else:
            raise ValueError(f"{log_base} is not supported.")

    def db_to_amp(self, x, log_base=10.0, multi_coef=1.0):
        if log_base is None:
            return np.power(np.e, (x) * (1.0/multi_coef))
        elif log_base == 10.0:
            return np.power(10.0, (x) * (1.0/multi_coef))
        elif log_base == 2.0:
            return np.power(2.0, (x) * (1.0/multi_coef))
        else:
            raise ValueError(f"{log_base} is not supported.")

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


def compute_mel_spectrogram(cfg, wav):
    # init 
    compute_mel_type = cfg.dataset.compute_mel_type
    compute_mel = ComputeMel(sampling_rate=cfg.dataset.sampling_rate,
                                    fft_size=cfg.dataset.fft_size,
                                    hop_size=cfg.dataset.hop_size,
                                    win_length=cfg.dataset.win_length,
                                    window=cfg.dataset.window,
                                    num_mels=cfg.dataset.num_mels,
                                    num_filts=cfg.dataset.num_filts,
                                    fmax=cfg.dataset.fmax, 
                                    fmin=cfg.dataset.fmin)

    # check
    assert compute_mel_type in ["fbank", "fbank_log", "fbank_nopreemphasis_log_manual", "compute_fbanks_preemphasis_log_manual", "pcen", "fbank_cpu"], \
        "[ERROR:] Audio compute mel type is wronge, please check"

    # compute mel
    if compute_mel_type == "fbank":
        mel = compute_mel.compute_fbanks(wav)
    elif compute_mel_type == "fbank_log":
        mel = compute_mel.compute_fbanks_log(wav)
    elif compute_mel_type == "fbank_nopreemphasis_log_manual":
        mel = compute_mel.fbank_nopreemphasis_log_manual(wav)
    elif compute_mel_type == "fbank_preemphasis_log_manual":
        mel = compute_mel.compute_fbanks_preemphasis_log_manual(wav)
    elif compute_mel_type == "pcen":
        mel = compute_mel.compute_pcen(wav)
    elif compute_mel_type == "fbank_cpu":
        if cfg.dataset.augmentation.on and cfg.dataset.augmentation.vtlp_on:
            mel = compute_mel.compute_fbanks_cpu(wav, cfg.dataset.augmentation.vtlp_on)
        else:
            mel = compute_mel.compute_fbanks_cpu(wav)
    return mel.astype(np.float32)


def compute_inv_mel_spectrogram(cfg, mel):
    """Converts mel spectrogram to waveform using librosa"""
    # init 
    compute_mel_type = cfg.dataset.compute_mel_type
    assert compute_mel_type == "fbank_preemphasis_log_manual", "only support audio compute mel type: fbank_preemphasis_log_manual"

    compute_mel = ComputeMel(sampling_rate=cfg.dataset.sampling_rate,
                                    fft_size=cfg.dataset.fft_size,
                                    hop_size=cfg.dataset.hop_size,
                                    win_length=cfg.dataset.win_length,
                                    window=cfg.dataset.window,
                                    num_mels=cfg.dataset.num_mels,
                                    num_filts=cfg.dataset.num_filts,
                                    fmax=cfg.dataset.fmax, 
                                    fmin=cfg.dataset.fmin)

    # denormalize
    if hparams.signal_normalization:
        D = compute_mel.denormalize(mel)
    else:
        D = mel
    
    # mel to linear
    S = compute_mel.mel_to_linear(compute_mel.db_to_amp(D + hparams.ref_level_db, multi_coef=20))  # Convert back to linear
    
    # griffin_lim
    return compute_mel.inv_preemphasis(compute_mel.griffin_lim(S ** hparams.power), hparams.preemphasis, hparams.preemphasize)


def compute_pre_emphasis(data):
    return ComputeMel.preemphasis(data, hparams.preemphasis, hparams.preemphasize)

def compute_de_emphasis(data):
    return ComputeMel.inv_preemphasis(data, hparams.preemphasis, hparams.preemphasize)