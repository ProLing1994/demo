from multiprocessing import Manager
import os
import sys
import random
import torch
from torch.utils.data import Dataset, DataLoader


sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
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
        assert len(self.data_list_filter) != 0, f"Not found any audio files."

        # speaker_id
        self.speaker_list = list(set(self.data_pd['speaker'].to_list()))
        self.speaker_list.sort()
        self.speaker_dict = {self.speaker_list[idx] : idx for idx in range(len(self.speaker_list))}

        print("Found {} samples, {} speakers. ".format(len(self.data_list), len(self.speaker_list)))
        
        if self.allow_cache:
            # NOTE: Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.data_list_filter))]

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        if not hasattr(self, 'lmdb_dict'):
            self.lmdb_dict = load_lmdb(self.cfg, self.mode)

        lmdb_dataset = self.data_pd.loc[index, 'dataset']
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
        wav = read_audio_lmdb(self.lmdb_dict[lmdb_dataset], str(data_name))

        # data augmentation
        if self.cfg.dataset.augmentation.on and self.cfg.dataset.augmentation.longer_senteces_on:
            text, wav = dataset_augmentation.dataset_augmentation_longer_senteces(self.cfg, text, wav, data_by_spk, self.lmdb_dict)

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

        mel = audio.compute_mel_spectrogram(self.cfg, wav).T.astype(np.float32)

        # speaker
        speaker_id = self.speaker_dict[speaker]

        # embed wav
        speaker_pd = self.data_pd[self.data_pd['speaker'] == speaker]
        speaker_data_name = random.choice(speaker_pd['unique_utterance'].to_list()) 
        embed_wav = read_audio_lmdb(self.lmdb_dict[lmdb_dataset], str(speaker_data_name))
        return text, mel, speaker_id, embed_wav

    def filter_by_threshold(self):
        self.data_list = []
        self.data_list_filter = []
        audio_length_threshold = int(self.cfg.dataset.sampling_rate * self.cfg.dataset.clip_duration_ms / 1000)

        for index, row in self.data_pd.iterrows():
            dataset_name = self.data_pd.loc[index, 'dataset']
            data_name = self.data_pd.loc[index, 'unique_utterance']

            # wav
            wav_path = os.path.join(self.cfg.general.data_dir, 'dataset_audio_hdf5', dataset_name, data_name.split('.')[0] + '.h5')
            wav = read_hdf5(wav_path, "wave")
            wav_length = len(wav)

            if wav_length > audio_length_threshold:
                self.data_list_filter.append(index)

            self.data_list.append(index)

        if len(self.data_list) != len(self.data_list_filter):
            print(f"[Warning] Some files are filtered by audio length threshold ({len(self.data_list)} -> {len(self.data_list_filter)}).")


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