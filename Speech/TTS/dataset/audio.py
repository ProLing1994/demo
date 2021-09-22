import glob
import librosa
import numpy as np
import os
import pcen
import sys
import struct
from scipy.ndimage.morphology import binary_dilation
from scipy.io import wavfile
from typing import Optional, Union
import torch
import webrtcvad

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/TTS')
from config.hparams import *
from dataset import logmmse

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_feature_pyimpl import Feature

def preprocess_speaker(cfg, data_files, row, no_alignments=False):
    file_path = row['file']
    metadata = []
    if no_alignments:
        # TO DO
        pass
    else:
        # Process alignment file (LibriSpeech support)
        # Gather the utterance audios and texts
        folder_path = os.path.dirname(file_path)
        file_name = str(os.path.basename(file_path)).split('.')[0]
        alignments_fpath = glob.glob(os.path.join(folder_path, "*.alignment.txt"))[0]
        with open(alignments_fpath, "r") as alignments_file:
            alignments = [line.rstrip().split(" ") for line in alignments_file]

        # Iterate over each entry in the alignments file
        for wav_fname, words, end_times in alignments:
            if wav_fname != file_name:
                continue

            wav_fpath = os.path.join(folder_path, wav_fname + ".flac")
            words = words.replace("\"", "").split(",")
            end_times = list(map(float, end_times.replace("\"", "").split(",")))

            # Process each sub-utterance
            wavs, texts = split_on_silences(cfg, wav_fpath, words, end_times)
            for i, (wav, text) in enumerate(zip(wavs, texts)):
                sub_basename = "%s_%02d" % (wav_fname, i)
                metadata.append(process_utterance(cfg, wav, text, data_files, row, sub_basename))
    return [m for m in metadata if m is not None]


def split_on_silences(cfg, wav_fpath, words, end_times):
    # Load the audio waveform
    wav, _ = librosa.load(str(wav_fpath), cfg.dataset.sample_rate)
    if rescale:
        wav = wav / np.abs(wav).max() * rescaling_max
    
    words = np.array(words)
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    assert len(words) == len(end_times) == len(start_times)
    assert words[0] == "" and words[-1] == ""
    
    # Find pauses that are too long
    mask = (words == "") & (end_times - start_times >= silence_min_duration_split)
    mask[0] = mask[-1] = True
    breaks = np.where(mask)[0]

    # Profile the noise from the silences and perform noise reduction on the waveform
    silence_times = [[start_times[i], end_times[i]] for i in breaks]
    silence_times = (np.array(silence_times) * cfg.dataset.sample_rate).astype(np.int)
    noisy_wav = np.concatenate([wav[stime[0]:stime[1]] for stime in silence_times])
    if len(noisy_wav) > cfg.dataset.sample_rate * 0.02:
        profile = logmmse.profile_noise(noisy_wav, cfg.dataset.sample_rate)
        wav = logmmse.denoise(wav, profile, eta=0)
    
    # Re-attach segments that are too short
    segments = list(zip(breaks[:-1], breaks[1:]))
    segment_durations = [start_times[end] - end_times[start] for start, end in segments]
    i = 0
    while i < len(segments) and len(segments) > 1:
        if segment_durations[i] < utterance_min_duration:
            # See if the segment can be re-attached with the right or the left segment
            left_duration = float("inf") if i == 0 else segment_durations[i - 1]
            right_duration = float("inf") if i == len(segments) - 1 else segment_durations[i + 1]
            joined_duration = segment_durations[i] + min(left_duration, right_duration)

            # Do not re-attach if it causes the joined utterance to be too long
            if joined_duration > cfg.dataset.window_size_ms * max_mel_frames / 1000:
                i += 1
                continue

            # Re-attach the segment with the neighbour of shortest duration
            j = i - 1 if left_duration <= right_duration else i
            segments[j] = (segments[j][0], segments[j + 1][1])
            segment_durations[j] = joined_duration
            del segments[j + 1], segment_durations[j + 1]
        else:
            i += 1
    
    # Split the utterance
    segment_times = [[end_times[start], start_times[end]] for start, end in segments]
    segment_times = (np.array(segment_times) * cfg.dataset.sample_rate).astype(np.int)
    wavs = [wav[segment_time[0]:segment_time[1]] for segment_time in segment_times]
    texts = [" ".join(words[start + 1:end]).replace("  ", " ") for start, end in segments]
    
    return wavs, texts


def process_utterance(cfg, wav: np.ndarray, text: str, data_files: list, row, 
                      sub_basename: str):
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume  
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.

    # Trim silence
    wav = preprocess_wav(wav, cfg.dataset.sample_rate, normalize=normalize, trim_silence=trim_silence)
    
    # Skip utterances that are too short
    if len(wav) < utterance_min_duration * cfg.dataset.sample_rate:
        return None
    
    # Compute the mel spectrogram
    mel_spectrogram = audio_preprocess(cfg, wav)
    mel_frames = mel_spectrogram.shape[0]
    
    # Skip utterances that are too long
    if mel_frames > max_mel_frames and clip_mels_length:
        return None
    
    # Return a dataset list describing this training example
    data_files.append({'dataset': row['dataset'], 'speaker': row['speaker'], 'section': row['section'], \
                        'utterance': row['utterance'], 'file': row['file'], 'text': text, \
                        'sub_basename': sub_basename, 'mode': row['mode']})
    return sub_basename, wav
    

def preprocess_wav(fpath_or_wav: Union[str, np.ndarray],
                   sampling_rate: Union[int],
                   source_sr: Optional[int] = None,
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
    if source_sr is not None and source_sr != sampling_rate:
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
    return audio_data.astype(np.float32)


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