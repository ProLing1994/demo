
import librosa
import numpy as np
import os
import pandas as pd
import random
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech/')
from Basic.config import hparams
from Basic.dataset import audio 
from Basic.dataset import dataset_augmentation
from Basic.utils import lmdb_tools

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

    def speaker_name(self):
        return self.speaker_name

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
    def __init__(self, cfg, mode, bool_trainning=True):
        # init
        self.cfg = cfg
        self.mode = mode
        self.bool_trainning = bool_trainning
        self.data_pd = load_data_pd(cfg, mode)
        self.speaker_list = list(set(self.data_pd['speaker'].to_list()))
        self.speaker_dict = {self.speaker_list[idx] : idx for idx in range(len(self.speaker_list))}
        if len(self.speaker_list) == 0:
            raise Exception("No speakers found. ")

        self.speakers = [Speaker(cfg, self.data_pd, speaker_name) \
                            for speaker_name in self.speaker_list]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        return len(self.speaker_list)
        
    def __getitem__(self, index):
        if not hasattr(self, 'lmdb_dict'):
            self.lmdb_dict = load_lmdb(self.cfg, self.mode)

        if not hasattr(self, 'background_data'):
            self.background_data = load_background_data(self.cfg)

        speakers = next(self.speaker_cycler)
        utterances = speakers.random_partial(self.cfg.train.utterances_per_speaker)
        assert len(utterances) == self.cfg.train.utterances_per_speaker

        # Array of shape (n_utterances, n_frames, mel_n), e.g. for 1 speakers with
        # 10 utterances each of 160 frames of 40 mel coefficients: (10, 160, 40)
        mel = np.array(gen_data(self.cfg, self.lmdb_dict, self.background_data, utterances, self.bool_trainning))

        # speaker
        speaker_name = speakers.speaker_name
        label = np.array([self.speaker_dict[speaker_name]] * len(mel))
        return mel, label


class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, cfg, dataset, speakers_per_batch, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=True,
                 timeout=0, worker_init_fn=None, bool_trainning=True):
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
        self.cfg = cfg
        self.bool_trainning = bool_trainning

    def collate(self, data):
        return data_batch(self.cfg, self.bool_trainning, data) 


def data_batch(cfg, bool_trainning, data):
    # Array of shape (n_speakers * n_utterances, n_frames, mel_n), e.g. for 2 speakers with
    # 10 utterances each of 160 frames of 40 mel coefficients: (20, 160, 40)
    mel = np.array([frames for speaker_data in data for frames in speaker_data[0]])
    label = np.array([frames for speaker_data in data for frames in speaker_data[1]])
    assert len(mel) == len(label)

    # data dynamic length
    if cfg.dataset.clip_duration_dynamic_length_on and bool_trainning:
        # 由于模型的特殊性，输入音频数据可以是变长的
        data_frames = mel.shape[1]
        min_frames = int(cfg.dataset.clip_duration_dynamic_ratio[0] * data_frames)
        max_frames = int(cfg.dataset.clip_duration_dynamic_ratio[1] * data_frames)
        
        real_frames = np.random.randint(min_frames, max_frames)  
        mel = mel[:, :real_frames, :]

    # Convert all to tensor
    mel = torch.from_numpy(mel).float()
    label = torch.from_numpy(label)
    return mel, label


def load_data_pd(cfg, mode):
    # load data_pd
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]
        csv_path = os.path.join(cfg.general.data_dir, dataset_name + '.csv')
    
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
        lmdb_path = os.path.join(cfg.general.data_dir, 'dataset_audio_lmdb', '{}.lmdb'.format(dataset_name+'_'+mode))
        if not os.path.exists(lmdb_path):
            print("[Warning] data do not exists: {}".format(lmdb_path))
            continue
        lmdb_env = lmdb_tools.load_lmdb_env(lmdb_path)
        lmdb_dict[dataset_name] = lmdb_env

    return lmdb_dict


def load_background_data(cfg):
    # load background_data
    background_data_pd = pd.read_csv(os.path.join(cfg.general.data_dir, 'background_noise_files.csv'))
    background_data_lmdb_path = os.path.join(cfg.general.data_dir, 'dataset_audio_lmdb', '{}.lmdb'.format(hparams.BACKGROUND_NOISE_DIR_NAME))
    background_data = []
    background_data_lmdb_env = lmdb_tools.load_lmdb_env(background_data_lmdb_path)
    for _, row in background_data_pd.iterrows():
        background_data.append(lmdb_tools.read_audio_lmdb(background_data_lmdb_env, row.file))
    
    return background_data

def gen_data(cfg, lmdb_dict, background_data, utterances, bool_trainning=True):
    mel_list = []

    for utterance_id in range(len(utterances)):
        utterance_pd = utterances[utterance_id]
        wav = lmdb_tools.read_audio_lmdb(lmdb_dict[str(utterance_pd['dataset'].values[0])], str(utterance_pd['file'].values[0]))
        assert len(wav) > 0, "{} {}".format(str(utterance_pd['dataset'].values[0]), str(utterance_pd['file'].values[0]))

        # data trim_silence
        if hparams.trim_silence:
            wav = audio.trim_silence(wav)

        # data alignment
        wav = dataset_augmentation.dataset_alignment(cfg, wav, bool_replicate=True)

        # data rescale
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max

        # data augmentation
        if cfg.dataset.augmentation.on and bool_trainning:
            wav = dataset_augmentation.dataset_augmentation_waveform(cfg, wav, background_data)

        # Compute the mel spectrogram
        mel = audio.compute_mel_spectrogram(cfg, wav)

        # data augmentation
        if cfg.dataset.augmentation.on and cfg.dataset.augmentation.spec_on and bool_trainning:
            mel = dataset_augmentation.dataset_augmentation_spectrum(cfg, mel)
        
        mel_list.append(mel)
    return mel_list