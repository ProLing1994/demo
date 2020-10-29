import numpy as np
import os
import pandas as pd 
import pickle
import sys
import time
import torch 

from torch.utils.data import Dataset

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from dataset.kws.dataset_helper import *

class SpeechDataset(Dataset):
  """
  Training dataset for Key word spotting
  """
  def __init__(self, cfg, mode, augmentation_on=True):
    # init
    super().__init__()

    # data index
    self.label_index = load_label_index(cfg.dataset.label.positive_label)

    # load data 
    data_pd = pd.read_csv(cfg.general.data_csv_path)

    self.mode_type = mode
    self.data_pd = data_pd[data_pd['mode'] == mode]
    self.input_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date), 'dataset', mode)
    self.data_file_list = self.data_pd['file'].tolist()
    self.data_mode_list = self.data_pd['mode'].tolist()
    self.data_label_list = self.data_pd['label'].tolist()

    self.input_channel = cfg.dataset.input_channel
    self.data_size_h = cfg.dataset.data_size[1]
    self.data_size_w = cfg.dataset.data_size[0]

  def __len__(self):
    """ get the number of images in this dataset """
    return len(self.data_file_list)

  def __getitem__(self, index):
    """ get the item """
    # record time
    # begin_t = time.time() 

    audio_file = self.data_file_list[index]
    audio_mode = self.data_mode_list[index]
    audio_label = self.data_label_list[index]
    assert audio_mode == self.mode_type, "[ERROR:] Something wronge about mode, please check"

    # load label idx
    audio_label_idx = self.label_index[audio_label]

    # load data
    input_dir = os.path.join(self.input_dir, audio_label)
    data = load_preload_audio(audio_file, index, audio_label, audio_label_idx, input_dir)
    
    # print('Load data Time: {}'.format((time.time() - begin_t) * 1.0))

    # To tensor
    data_tensor = torch.from_numpy(data.reshape(1, -1, 40))
    data_tensor = data_tensor.float()
    label_tensor = torch.tensor(audio_label_idx)

    # check tensor
    assert data_tensor.shape[0] == self.input_channel
    assert data_tensor.shape[1] == self.data_size_h
    assert data_tensor.shape[2] == self.data_size_w
    return data_tensor, label_tensor, index
    
    

