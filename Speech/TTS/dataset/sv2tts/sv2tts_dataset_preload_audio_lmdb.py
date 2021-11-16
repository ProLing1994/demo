import os
from re import X
import pandas as pd
import sys
import random
import torch
from torch.utils.data import Dataset, DataLoader


sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.config import hparams
from Basic.dataset import audio
from Basic.utils.lmdb_tools import *

from TTS.config.sv2tts.hparams import *
from TTS.dataset.text.text import *
from TTS.dataset.sv2tts.audio import *


def load_data_pd(cfg, mode):
    # load data_pd
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]
        csv_path = os.path.join(cfg.general.data_dir, dataset_name + '_' + mode + '.csv')

        if not os.path.exists(csv_path):
            print("[Warning] csv path do not exist: {}".format(csv_path))
            continue

        data_pd_temp = pd.read_csv(csv_path)
        if dataset_idx == 0:
            data_pd = data_pd_temp
        else:
            data_pd = pd.concat([data_pd, data_pd_temp])

    data_pd = data_pd[data_pd["mode"] == mode]
    data_pd.reset_index(drop=True, inplace=True) 
    return data_pd


def load_lmdb(cfg, mode):
    # load lmdb_dict
    lmdb_dict = {}
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]
        # lmdb
        lmdb_path = os.path.join(cfg.general.data_dir, 'dataset_audio_lmdb', '{}.lmdb'.format(dataset_name+'_'+mode))

        if not os.path.exists(lmdb_path):
            print("[Warning] data do not exists: {}".format(lmdb_path))
            continue

        lmdb_env = load_lmdb_env(lmdb_path)
        lmdb_dict[dataset_name] = lmdb_env

    return lmdb_dict


def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)


def pad2d(x, max_len, pad_value=0):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant", constant_values=pad_value)


class SynthesizerDataset(Dataset):
    def __init__(self, cfg, mode, augmentation_on=True):
        # init
        self.cfg = cfg
        self.mode = mode
        self.augmentation_on = augmentation_on

        self.data_pd = load_data_pd(cfg, mode)
        self.data_list = self.data_pd['unique_utterance'].to_list()

        self.speaker_list = list(set(self.data_pd['speaker'].to_list()))
        self.speaker_list.sort()
        self.speaker_dict = {self.speaker_list[idx] : idx for idx in range(len(self.speaker_list))}

        if len(self.data_list) == 0:
            raise Exception("No speakers found. ")

        print("Found {} samples, {} speakers. ".format(len(self.data_list), len(self.speaker_list)))

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        if not hasattr(self, 'lmdb_dict'):
            self.lmdb_dict = load_lmdb(self.cfg, self.mode)

        lmdb_dataset = self.data_pd.loc[index, 'dataset']
        data_name = self.data_pd.loc[index, 'unique_utterance']
        speaker = self.data_pd.loc[index, 'speaker']

        if self.cfg.dataset.language == 'chinese':
            text = self.data_pd.loc[index, self.cfg.dataset.symbols]
        elif self.cfg.dataset.language == 'english':
            text = self.data_pd.loc[index, 'text']
        else:
            raise Exception("[ERROR:] Unknow dataset language: {}".format(self.cfg.dataset.language))

        # text
        # Get the text and clean it
        text = text_to_sequence(text, self.cfg.dataset.tts_cleaner_names, lang=self.cfg.dataset.symbols_lang)
        
        # Convert the list returned by text_to_sequence to a numpy array
        text = np.asarray(text).astype(np.int32)

        # mel
        wav = read_audio_lmdb(self.lmdb_dict[lmdb_dataset], str(data_name))
        mel = audio.compute_mel_spectrogram(self.cfg, wav).T.astype(np.float32)

        # speaker
        speaker_id = self.speaker_dict[speaker]

        # embed wav
        speaker_pd = self.data_pd[self.data_pd['speaker'] == speaker]
        speaker_data_name = random.choice(speaker_pd['unique_utterance'].to_list()) 
        embed_wav = read_audio_lmdb(self.lmdb_dict[lmdb_dataset], str(speaker_data_name))
        return text, mel, speaker_id, embed_wav


class SynthesizerDataLoader(DataLoader):
    def __init__(self, dataset, sampler, cfg):
        super().__init__(
            dataset=dataset, 
            batch_size=cfg.train.batch_size, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=None, 
            num_workers=cfg.train.num_threads,
            collate_fn=lambda data: self.collate_synthesizer(data, cfg),
            pin_memory=True, 
            drop_last=False, 
            timeout=0, 
            worker_init_fn=None
        )

    def collate_synthesizer(self, data, cfg):
        # Sort 
        data = sorted(data, key=lambda x: len(x[0]), reverse=True)

        # Text
        char_lengths = [len(x[0]) for x in data]
        max_char_length = max(char_lengths)

        chars = [pad1d(x[0], max_char_length) for x in data]
        chars = np.stack(chars)         # shape: [b, text_t]

        # Mel spectrogram
        mel_lengths = [x[1].shape[1] for x in data]
        for idx in range(len(mel_lengths)):
            if mel_lengths[idx] % cfg.net.r != 0:
                mel_lengths[idx] += cfg.net.r - mel_lengths[idx] % cfg.net.r
        max_mel_length = max(mel_lengths)

        # WaveRNN mel spectrograms are normalized to [0, 1] so zero padding adds silence
        # By default, SV2TTS uses symmetric mels, where -1*max_abs_value is silence.
        if hparams.symmetric_mels:
            mel_pad_value = -1 * hparams.max_abs_value
        else:
            mel_pad_value = 0

        mels = [pad2d(x[1], max_mel_length, pad_value=mel_pad_value) for x in data]
        mels = np.stack(mels)           # shape: [b, mel_f, mel_t]
        
        # Mel Frames: stop
        stops = torch.zeros(mels.shape[0], mels.shape[-1])
        for j, k in enumerate(mel_lengths):
            stops[j, int(k)-1:] = 1     # shape: [b, mel_t]

        # Speaker id (mutil speaker)
        speaker_ids = [x[2] for x in data]
        speaker_ids = np.stack(speaker_ids)
        
        # Speaker embedding (SV2TTS)
        embed_wavs = [x[3] for x in data]

        # Convert all to tensor
        chars = torch.tensor(chars).long()
        char_lengths = torch.tensor(char_lengths)
        mels = torch.tensor(mels)
        mel_lengths = torch.tensor(mel_lengths)
        speaker_ids = torch.tensor(speaker_ids)
        return chars, char_lengths, mels, mel_lengths, stops, speaker_ids, embed_wavs