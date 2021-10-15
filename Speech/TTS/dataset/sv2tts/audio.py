import glob
import librosa
import numpy as np
import os
import sys
from scipy.ndimage.morphology import binary_dilation

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.config import hparams
from Basic.dataset import logmmse
from Basic.dataset import audio

from TTS.config.sv2tts import hparams as hparams_tts


def preprocess_speaker_librispeech(cfg, data_files, row):
    # init 
    metadata = []
    file_path = row['file']

    folder_path = os.path.dirname(file_path)
    file_name = str(os.path.basename(file_path)).split('.')[0]

    # Process alignment file (LibriSpeech support)
    # Gather the utterance audios and texts
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

        # Load the audio waveform
        wav, _ = librosa.load(str(wav_fpath), cfg.dataset.sample_rate)

        # data rescale
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max

        # Process each sub-utterance
        wavs, texts = split_on_silences(cfg, wav_fpath, words, end_times)
        for i, (wav, text) in enumerate(zip(wavs, texts)):
            unique_utterance = "%s_%02d" % (wav_fname, i)
            metadata.append(process_utterance(cfg, wav, text, data_files, row, unique_utterance))
    return metadata


def preprocess_speaker_aishell3(cfg, data_files, row):
    # init 
    metadata = []
    wav_fpath = row['file']

    # load wav
    wav, _ = librosa.load(str(wav_fpath), cfg.dataset.sample_rate)

    # trim silence
    if hparams.trim_silence: 
        wav = audio.trim_silence(wav)

    # data rescale
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # text
    text = ''
    
    # unique_utterance 
    unique_utterance = row['utterance']

    metadata.append(process_utterance(cfg, wav, text, data_files, row, unique_utterance))
    return metadata


def preprocess_speaker(cfg, data_files, row, dataset_name):

    if dataset_name == 'librispeech_clean_360' or dataset_name == 'librispeech_clean_100' or dataset_name == 'librispeech_test_clean': 
        metadata = preprocess_speaker_librispeech(cfg, data_files, row)
    elif dataset_name == 'Aishell3': 
        metadata = preprocess_speaker_aishell3(cfg, data_files, row)
    return [m for m in metadata if m is not None]


def split_on_silences(cfg, wav, words, end_times):
    words = np.array(words)
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    assert len(words) == len(end_times) == len(start_times)
    assert words[0] == "" and words[-1] == ""
    
    # Find pauses that are too long
    mask = (words == "") & (end_times - start_times >= hparams_tts.silence_min_duration_split)
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
        if segment_durations[i] < hparams_tts.utterance_min_duration:
            # See if the segment can be re-attached with the right or the left segment
            left_duration = float("inf") if i == 0 else segment_durations[i - 1]
            right_duration = float("inf") if i == len(segments) - 1 else segment_durations[i + 1]
            joined_duration = segment_durations[i] + min(left_duration, right_duration)

            # Do not re-attach if it causes the joined utterance to be too long
            if joined_duration > cfg.dataset.window_size_ms * hparams_tts.max_mel_frames / 1000:
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
                      unique_utterance: str):
    # Skip utterances that are too short
    if len(wav) < hparams_tts.utterance_min_duration * cfg.dataset.sample_rate:
        return None

    # desired samples
    clip_duration_ms = hparams_tts.max_mel_frames * cfg.dataset.window_stride_ms
    desired_samples = int(cfg.dataset.sample_rate * clip_duration_ms / 1000)
    
    # Skip utterances that are too long
    if len(wav) > desired_samples and hparams_tts.clip_mels_length:
        return None
    
    # Return a dataset list describing this training example
    data_files.append({'dataset': row['dataset'], 'speaker': row['speaker'], 'section': row['section'], \
                        'utterance': row['utterance'], 'file': row['file'], 'text': text, \
                        'unique_utterance': unique_utterance, 'mode': row['mode']})
    return unique_utterance, wav
