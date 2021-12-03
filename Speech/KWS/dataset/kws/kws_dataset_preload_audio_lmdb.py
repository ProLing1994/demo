from multiprocessing import Manager
import numpy as np
import os
import sys
import pandas as pd
import torch

from torch.utils.data import Dataset

# sys.path.insert(0, '/yuanhuan/code/demo/Speech/')
sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio

from KWS.config.kws import hparams
from KWS.dataset.kws.dataset_helper import load_label_index
from KWS.dataset.kws import dataset_augmentation
from KWS.utils import lmdb_tools 


class SpeechDataset(Dataset):
    """
    Training dataset for Key word spotting
    """

    def __init__(self, cfg, mode, augmentation_on=True):
        # init
        super().__init__()

        self.cfg = cfg
        self.bool_trainning = augmentation_on
        self.allow_cache = self.cfg.dataset.allow_cache

        # data index
        self.positive_label_list = cfg.dataset.label.positive_label
        self.negative_label_list = cfg.dataset.label.negative_label
        self.positive_label_together = cfg.dataset.label.positive_label_together
        self.negative_label_together = cfg.dataset.label.negative_label_together

        if self.positive_label_together and self.negative_label_together:
            self.positive_label_together_label_list = cfg.dataset.label.positive_label_together_label
            self.negative_label_together_label_list = cfg.dataset.label.negative_label_together_label
            self.label_index = load_label_index(self.positive_label_together_label_list, self.negative_label_together_label_list)
        elif self.positive_label_together:
            self.positive_label_together_label_list = cfg.dataset.label.positive_label_together_label
            self.label_index = load_label_index(self.positive_label_together_label_list, cfg.dataset.label.negative_label)
        elif self.negative_label_together:
            self.negative_label_together_label_list = cfg.dataset.label.negative_label_together_label
            self.label_index = load_label_index(self.positive_label_list, self.negative_label_together_label_list)
        else:
            self.label_index = load_label_index(self.positive_label_list, cfg.dataset.label.negative_label)

        # load data pandas
        data_pd = pd.read_csv(cfg.general.data_csv_path, encoding='utf_8_sig')

        self.mode_type = mode
        self.data_pd = data_pd[data_pd['mode'] == mode]
        self.data_file_list = self.data_pd['file'].tolist()
        self.data_mode_list = self.data_pd['mode'].tolist()
        self.data_label_list = self.data_pd['label'].tolist()

        self.input_channel = cfg.dataset.input_channel
        self.data_size_h = cfg.dataset.data_size[1]
        self.data_size_w = cfg.dataset.data_size[0]
        
        if self.allow_cache:
            # NOTE: Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.data_file_list))]

    def __len__(self):
        """ get the number of images in this dataset """
        return len(self.data_file_list)

    def save_audio(self, data, audio_label, audio_file):
        out_folder = os.path.join(self.cfg.general.save_dir, self.mode_type + '_audio', audio_label)

        # mkdir
        if not os.path.isdir(out_folder):
            try:
                os.makedirs(out_folder)
            except:
                pass

        filename = os.path.basename(audio_file)
        if not str(filename).endswith('.wav'):
            filename = str(filename).split('.')[0] + '.wav'
        audio.save_wav(data.copy(), os.path.join(out_folder, filename), self.cfg.dataset.sampling_rate)

    def __getitem__(self, index):
        """ get the item """

        if not hasattr(self, 'lmdb_env'):
            self.lmdb_path = os.path.join(os.path.dirname(self.cfg.general.data_csv_path), 'dataset_audio_lmdb', '{}.lmdb'.format(self.mode_type))
            self.lmdb_env = lmdb_tools.load_lmdb_env(self.lmdb_path)

        if not hasattr(self, 'background_data'):
            self.background_data = lmdb_tools.load_background_noise_lmdb(self.cfg)

        audio_file = self.data_file_list[index]
        audio_label = self.data_label_list[index]
        audio_mode = self.data_mode_list[index]
        assert audio_mode == self.mode_type, "[ERROR:] Something wronge about mode, please check"

        # load label idx
        if self.positive_label_together and audio_label in self.positive_label_list:
            audio_label_idx = self.label_index[self.positive_label_together_label_list[0]]
        elif self.negative_label_together and audio_label in self.negative_label_list:
            audio_label_idx = self.label_index[self.negative_label_together_label_list[0]]
        else:
            audio_label_idx = self.label_index[audio_label]

        # load data
        if self.allow_cache and len(self.caches[index]) != 0:
            data = self.caches[index]
        else:
            data = lmdb_tools.read_audio_lmdb(self.lmdb_env, audio_file)
            assert len(data) != 0, "[ERROR:] Something wronge about load data, please check: {}".format(audio_file)

            if self.allow_cache:
                self.caches[index] = data
                
        # data augmentation
        if audio_label == hparams.SILENCE_LABEL:
            data = dataset_augmentation.dataset_add_noise(self.cfg, data, self.background_data, bool_force_add_noise=True)
        else:
            if self.cfg.dataset.augmentation.on and self.bool_trainning:
                if audio_label == hparams.UNKNOWN_WORD_LABEL:
                    data = dataset_augmentation.dataset_augmentation_waveform(self.cfg, data, self.background_data, bool_time_shift_multiple=True)
                else:
                    data = dataset_augmentation.dataset_augmentation_waveform(self.cfg, data, self.background_data)
            else:
                data = dataset_augmentation.dataset_alignment(self.cfg, data) 

        if self.cfg.debug.save_inputs:
            self.save_audio(data, audio_label, audio_file)

        # Compute the mel spectrogram
        data = audio.compute_mel_spectrogram(self.cfg, data)

        # data augmentation
        if self.cfg.dataset.augmentation.on and self.bool_trainning:
            data = dataset_augmentation.dataset_augmentation_spectrum(self.cfg, data)

        # To tensor
        data_tensor = torch.from_numpy(np.expand_dims(data, axis=0))
        data_tensor = data_tensor.float()
        label_tensor = torch.tensor(audio_label_idx)
        label_tensor = label_tensor.type(torch.LongTensor)

        # check tensor
        assert data_tensor.shape[0] == self.input_channel
        assert data_tensor.shape[1] == self.data_size_h
        assert data_tensor.shape[2] == self.data_size_w

        # print
        # print(audio_label, label_tensor)
        return data_tensor, label_tensor, index
