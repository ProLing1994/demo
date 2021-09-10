
import librosa
import numpy as np
import os
import pandas as pd
import random
import sys
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/SV')
from utils.lmdb_tools import *
from config.hparams import *
from dataset.audio import *

class RandomCycler:
    """
    Creates an internal copy of a sequence and allows access to its items in a constrained random 
    order. For a source sequence of n items and one or several consecutive queries of a total 
    of m items, the following guarantees hold (one implies the other):
        - Each item will be returned between m // n and ((m - 1) // n) + 1 times.
        - Between two appearances of the same item, there may be at most 2 * (n - 1) other items.
    """
    
    def __init__(self, source):
        if len(source) == 0:
            raise Exception("Can't create RandomCycler from an empty collection")
        self.all_items = list(source)
        self.next_items = []
    
    def sample(self, count: int):
        shuffle = lambda l: random.sample(l, len(l))
        
        out = []
        while count > 0:
            if count >= len(self.all_items):
                out.extend(shuffle(list(self.all_items)))
                count -= len(self.all_items)
                continue
            n = min(count, len(self.next_items))
            out.extend(self.next_items[:n])
            count -= n
            self.next_items = self.next_items[n:]
            if len(self.next_items) == 0:
                self.next_items = shuffle(list(self.all_items))
        return out
    
    def __next__(self):
        return self.sample(1)[0]


class Utterance:
    def __init__(self, cfg, data_pd, utterances_name):
        self.cfg = cfg
        self.data_pd = data_pd
        self.utterances_name = utterances_name

    def get_frames(self):
        return self.data_pd[self.data_pd["utterance"] == self.utterances_name]

    def random_partial(self):
        """
        Crops the frames into a partial utterance of n_frames
        
        :param n_frames: The number of frames of the partial utterance
        :return: the partial utterance frames and a tuple indicating the start and end of the 
        partial utterance in the complete utterance.
        """
        return self.get_frames()


class Speaker:
    def __init__(self, cfg, data_pd, speaker_name):
        self.cfg = cfg
        self.data_pd = data_pd
        self.speaker_name = speaker_name
        self.utterances = None
        self.utterance_cycler = None

    def _load_utterances(self):
        self.data_pd = self.data_pd[self.data_pd["speaker"] == self.speaker_name]
        self.utterances_list = list(set(self.data_pd['utterance'].to_list()))
        if len(self.utterances_list) == 0:
            raise Exception("No utterances found. ")

        self.utterances = [Utterance(self.cfg, self.data_pd, utterances_name) for utterances_name in self.utterances_list]
        self.utterance_cycler = RandomCycler(self.utterances)

    def random_partial(self, count):
        """
        Samples a batch of <count> unique partial utterances from the disk in a way that all 
        utterances come up at least once every two cycles and in a random order every time.
        
        :param count: The number of partial utterances to sample from the set of utterances from 
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than 
        the number of utterances available.
        """
        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)

        a = [u.random_partial() for u in utterances]

        return a


class SpeakerVerificationDataset(Dataset):
    def __init__(self, cfg, data_pd):
        # init
        self.speaker_list = list(set(data_pd['speaker'].to_list()))
        if len(self.speaker_list) == 0:
            raise Exception("No speakers found. ")

        self.speakers = [Speaker(cfg, data_pd, speaker_name) \
                            for speaker_name in self.speaker_list]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        return len(self.speaker_list)
        
    def __getitem__(self, index):
        return next(self.speaker_cycler)


class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, cfg, dataset, speakers_per_batch, utterances_per_speaker, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=True, timeout=0, 
                 worker_init_fn=None):
        self.cfg = cfg
        self.utterances_per_speaker = utterances_per_speaker

        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers):
        return speaker_batch(speakers, self.cfg, self.utterances_per_speaker) 


def speaker_batch(speakers, cfg, utterances_per_speaker):
    partials = {s: s.random_partial(utterances_per_speaker) for s in speakers}

    # Array of shape (n_speakers * n_utterances, n_frames, mel_n), e.g. for 3 speakers with
    # 4 utterances each of 160 frames of 40 mel coefficients: (12, 160, 40)
    data = np.array(gen_data(cfg, utterances_per_speaker, speakers, partials))
    return data


def open_lmdb(cfg):
    # load lmdb_dict
    lmdb_dict = {}
    for dataset_idx in range(len(cfg.general.TISV_dataset_list)):
        dataset_name = cfg.general.TISV_dataset_list[dataset_idx]
        # lmdb
        lmdb_path = os.path.join(cfg.general.data_dir, 'dataset_audio_lmdb', '{}.lmdb'.format(dataset_name))
        lmdb_env = load_lmdb_env(lmdb_path)
        lmdb_dict[dataset_name] = lmdb_env

    # init background_data
    background_data_pd = pd.read_csv(os.path.join(cfg.general.data_dir, 'background_noise_files.csv'))
    background_data_lmdb_path = os.path.join(cfg.general.data_dir, 'dataset_audio_lmdb', '{}.lmdb'.format(BACKGROUND_NOISE_DIR_NAME))
    background_data = []
    background_data_lmdb_env = load_lmdb_env(background_data_lmdb_path)
    for _, row in background_data_pd.iterrows():
        background_data.append(read_audio_lmdb(background_data_lmdb_env, row.file))
    
    return lmdb_dict, background_data


def gen_data(cfg, utterances_per_speaker, speakers, partials):
    lmdb_dict, background_data = open_lmdb(cfg)

    data_list = []
    for s in speakers:
        data_pd_list = partials[s]
        assert len(data_pd_list) == utterances_per_speaker

        for wav_id in range(len(data_pd_list)):
            data_pd = data_pd_list[wav_id]
            data = read_audio_lmdb(lmdb_dict[str(data_pd['dataset'].values[0])], str(data_pd['file'].values[0]))

            # data augmentation
            if cfg.dataset.augmentation.on:
                data = dataset_augmentation_waveform(cfg, data, background_data)
                pass
            else:
                data = dataset_alignment(cfg, data)

            # audio preprocess, get mfcc data
            data = audio_preprocess(cfg, data)

            # # data augmentation
            if cfg.dataset.augmentation.on and cfg.dataset.augmentation.spec_on:
                data = dataset_augmentation_spectrum(cfg, data)
            
            data_list.append(data)
    return data_list