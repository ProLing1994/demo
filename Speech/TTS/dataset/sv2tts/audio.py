import glob
import librosa
import numpy as np
import os
import sys
import struct
from scipy.ndimage.morphology import binary_dilation
import webrtcvad

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_feature_pyimpl import Feature

from Basic.config.hparams import *
from Basic.dataset import logmmse
from Basic.dataset.audio import *

from TTS.config.sv2tts.hparams import *


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
    mel_spectrogram = compute_mel_spectrogram(cfg, wav)
    mel_frames = mel_spectrogram.shape[0]
    
    # Skip utterances that are too long
    if mel_frames > max_mel_frames and clip_mels_length:
        return None
    
    # Return a dataset list describing this training example
    data_files.append({'dataset': row['dataset'], 'speaker': row['speaker'], 'section': row['section'], \
                        'utterance': row['utterance'], 'file': row['file'], 'text': text, \
                        'sub_basename': sub_basename, 'mode': row['mode']})
    return sub_basename, wav
