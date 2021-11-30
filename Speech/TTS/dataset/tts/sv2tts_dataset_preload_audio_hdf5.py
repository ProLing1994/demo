from multiprocessing import Manager
import os
import sys
import random
import torch
from torch.utils.data import Dataset, DataLoader


sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.config import hparams
from Basic.dataset import audio
from Basic.utils.hdf5_tools import *

from TTS.config.tts.hparams import *
from TTS.dataset.text.text import *
from TTS.dataset.tts.audio import *
from TTS.dataset.tts import dataset_augmentation
from TTS.dataset.tts.sv2tts_dataset_preload_audio_lmdb import load_data_pd, pad1d, pad2d


class SynthesizerDataset(Dataset):
    def __init__(self, cfg, mode, augmentation_on=True):
        # init
        self.cfg = cfg
        self.mode = mode
        self.augmentation_on = augmentation_on
        self.allow_cache = self.cfg.dataset.allow_cache

        self.data_pd = load_data_pd(cfg, mode)

        # filter by threshold
        self.filter_by_threshold()
        # assert the number of files
        assert len(self.data_pd) != 0, f"Not found any audio files."

        # speaker_id
        self.speaker_list = list(set(self.data_pd['speaker'].to_list()))
        self.speaker_list.sort()
        self.speaker_dict = {self.speaker_list[idx] : idx for idx in range(len(self.speaker_list))}

        print("Found {} samples, {} speakers. ".format(len(self.data_pd), len(self.speaker_list)))
        
        if self.allow_cache:
            # NOTE: Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.data_pd))]

    def __len__(self):
        return len(self.data_pd)
    
    def __getitem__(self, index):

        if self.allow_cache and len(self.caches[index]) != 0:
            return self.caches[index][0], self.caches[index][1], self.caches[index][2], self.caches[index][3]

        dataset_name = self.data_pd.loc[index, 'dataset']
        data_name = self.data_pd.loc[index, 'unique_utterance']
        speaker = self.data_pd.loc[index, 'speaker']
        data_by_spk = self.data_pd[self.data_pd['speaker'] == speaker]

        # text
        if self.cfg.dataset.language == 'chinese':
            text = self.data_pd.loc[index, self.cfg.dataset.symbols]
            text = str(text).strip()
            assert text[-1] in [',', '.', '/', '1', '2', '3', '4', '5'], "[ERROR:] text: {} | {} ".format(text, text[-1])
        elif self.cfg.dataset.language == 'english':
            text = self.data_pd.loc[index, 'text']
        else:
            raise Exception("[ERROR:] Unknow dataset language: {}".format(self.cfg.dataset.language))

        # mel
        # data augmentation
        if self.cfg.dataset.augmentation.on and self.cfg.dataset.augmentation.longer_senteces_on:
            wav_path = os.path.join(self.cfg.general.data_dir, 'dataset_audio_hdf5', dataset_name, data_name.split('.')[0] + '.h5')
            wav = read_hdf5(wav_path, "wave")
            text, wav = dataset_augmentation.dataset_augmentation_longer_senteces_hdf5(self.cfg, text, wav, data_by_spk)
            mel = audio.compute_mel_spectrogram(self.cfg, wav).T.astype(np.float32)
        else:
            wav_path = os.path.join(self.cfg.general.data_dir, 'dataset_audio_hdf5', dataset_name, data_name.split('.')[0] + '.h5')
            mel = read_hdf5(wav_path, "feats").T.astype(np.float32)
            # mel = read_hdf5(wav_path, "feats")

        # 中文句末需要变换为句号 . 
        if self.cfg.dataset.language == 'chinese':
            text = re.sub(',$', '.', text) 
            text = re.sub('.$', '.', text) 
            text = re.sub('/$', '.', text) 
            text = re.sub('1$', '1.', text) 
            text = re.sub('2$', '2.', text) 
            text = re.sub('3$', '3.', text) 
            text = re.sub('4$', '4.', text) 
            text = re.sub('5$', '5.', text) 
            assert text[-1] == '.', "[ERROR:] text: {} | {} ".format(text, text[-1])

        # Get the text and clean it
        text = text_to_sequence(text, self.cfg.dataset.tts_cleaner_names, lang=self.cfg.dataset.symbols_lang)
        
        # Convert the list returned by text_to_sequence to a numpy array
        text = np.asarray(text).astype(np.int32)

        # speaker
        speaker_id = self.speaker_dict[speaker]

        # embed wav
        speaker_pd = self.data_pd[self.data_pd['speaker'] == speaker]
        speaker_data_name = random.choice(speaker_pd['unique_utterance'].to_list()) 
        embed_wav_path = os.path.join(self.cfg.general.data_dir, 'dataset_audio_hdf5', dataset_name, speaker_data_name.split('.')[0] + '.h5')
        embed_wav = read_hdf5(embed_wav_path, "wave")

        if self.allow_cache:
            self.caches[index] = (text, mel, speaker_id, embed_wav)

        return text, mel, speaker_id, embed_wav

    def filter_by_threshold(self):
        self.drop_index_list = []
        audio_length_threshold = int(self.cfg.dataset.sampling_rate * self.cfg.dataset.clip_duration_ms / 1000)

        for index, row in self.data_pd.iterrows():
            dataset_name = self.data_pd.loc[index, 'dataset']
            data_name = self.data_pd.loc[index, 'unique_utterance']

            # wav
            wav_path = os.path.join(self.cfg.general.data_dir, 'dataset_audio_hdf5', dataset_name, data_name.split('.')[0] + '.h5')
            wav = read_hdf5(wav_path, "wave")
            wav_length = len(wav)

            if wav_length < audio_length_threshold:
                self.drop_index_list.append(index)

        if len(self.drop_index_list) != 0:
            print(f"[Warning] Some files are filtered by audio length threshold ({len(self.data_pd)} -> {len(self.data_pd) - len(self.drop_index_list)}).")
        
        # drop
        self.data_pd.drop(self.drop_index_list, inplace=True)
        self.data_pd.reset_index(drop=True, inplace=True)


class SynthesizerCollater(object):
    """Customized collater for Pytorch DataLoader in training."""
    def __init__(self, cfg):

        self.cfg = cfg

    def __call__(self, batch):
        # Sort 
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

        # Text
        char_lengths = [len(x[0]) for x in batch]
        max_char_length = max(char_lengths)

        chars = [pad1d(x[0], max_char_length) for x in batch]
        chars = np.stack(chars)         # shape: [b, text_t]

        # Mel spectrogram
        mel_lengths = [x[1].shape[1] for x in batch]
        max_mel_length = max(mel_lengths)

        mel_pad_value = 0
        mels = [pad2d(x[1], max_mel_length, pad_value=mel_pad_value) for x in batch]
        mels = np.stack(mels)           # shape: [b, mel_f, mel_t]
        
        # Mel Frames: stop
        stops = torch.zeros(mels.shape[0], mels.shape[-1])
        for j, k in enumerate(mel_lengths):
            stops[j, int(k)-1:] = 1     # shape: [b, mel_t]

        # Speaker id (mutil speaker)
        speaker_ids = [x[2] for x in batch]
        speaker_ids = np.stack(speaker_ids)
        
        # Speaker embedding (SV2TTS)
        embed_wavs = [x[3] for x in batch]

        # Convert all to tensor
        chars = torch.tensor(chars).long()
        char_lengths = torch.tensor(char_lengths)
        mels = torch.tensor(mels)
        mel_lengths = torch.tensor(mel_lengths)
        speaker_ids = torch.tensor(speaker_ids)
        return chars, char_lengths, mels, mel_lengths, stops, speaker_ids, embed_wavs