import librosa
import numpy as np
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio
from Basic.utils.hdf5_tools import *

def preprocess_audio_normal(cfg, row, hdf5_dir, data_lists):
    # init 
    wav_fpath = row['file']
    fft_size = cfg.dataset.fft_size
    hop_size = cfg.dataset.hop_size

    # load wav
    wav, _ = librosa.load(str(wav_fpath), cfg.dataset.sampling_rate)

    # trim silence
    if cfg.dataset.trim_silence: 
        wav = audio.trim_silence(wav, 
                                cfg.dataset.trim_threshold_in_db,
                                cfg.dataset.trim_frame_size,
                                cfg.dataset.trim_hop_size)

    mel = audio.compute_mel_spectrogram(cfg, wav)

    # make sure the audio length and feature length are matched
    wav = np.pad(wav, (0, fft_size), mode = "reflect")
    wav = wav[: len(mel) * hop_size]
    assert len(mel) * hop_size == len(wav)

    # text
    text = ''
    
    # unique_utterance 
    unique_utterance = row['utterance']

    data_lists.append({'dataset': row['dataset'], 'speaker': row['speaker'], 'section': row['section'], \
                        'utterance': row['utterance'], 'file': row['file'], 'text': text, \
                        'unique_utterance': unique_utterance, 'mode': row['mode']})

    save_name = unique_utterance.split('.')[0]
    write_hdf5(
        os.path.join(hdf5_dir, f"{save_name}.h5"),
        "wave",
        wav.astype(np.float32),
    )
    write_hdf5(
        os.path.join(hdf5_dir, f"{save_name}.h5"),
        "feats",
        mel.astype(np.float32),
    )

    return


def preprocess_audio_hdf5(cfg, row, dataset_name, hdf5_dir, data_lists):
    
    if dataset_name in ['librispeech_clean_360', 'librispeech_clean_100', 'librispeech_test_clean']: 
        # preprocess_audio_librispeech(cfg, data_files, row)
        raise NotImplemented
    elif dataset_name in ['Aishell3', 'BZNSYP', 'BwcKeyword']: 
        preprocess_audio_normal(cfg, row, hdf5_dir, data_lists)
    return 