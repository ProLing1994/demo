import numpy as np
import os
import pandas as pd 
import pickle
import time
import torch 

from torch.utils.data import Dataset

SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'

class SpeechDataset(Dataset):
  """
  Training dataset for Key word spotting
  """
  def __init__(self, cfg, mode, augmentation_on=True):
    # init
    super().__init__()
    data_path = cfg.general.data_path
    self.input_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date), 'dataset', mode)

    # data index
    self.positive_words_index = {}
    for index, positive_word in enumerate(cfg.dataset.label.positive_label):
      self.positive_words_index[positive_word] = index + 2
    self.positive_words_index.update({SILENCE_LABEL:SILENCE_INDEX, UNKNOWN_WORD_LABEL:UNKNOWN_WORD_INDEX})

    # load data 
    data_pd = pd.read_csv(data_path)
    data_mode_pd = data_pd[data_pd['mode'] == mode]

    self.mode_type = mode
    self.data_mode_pd = data_mode_pd
    self.data_mode_pd_file = self.data_mode_pd['file'].tolist()
    self.data_mode_pd_mode = self.data_mode_pd['mode'].tolist()
    self.data_mode_pd_label = self.data_mode_pd['label'].tolist()

    self.input_channel = cfg.dataset.input_channel
    self.data_size_h = cfg.dataset.data_size[1]
    self.data_size_w = cfg.dataset.data_size[0]

  def __len__(self):
    """ get the number of images in this dataset """
    return len(self.data_mode_pd_file)

  def __getitem__(self, index):
    """ get the item """
    # record time
    # begin_t = time.time() 

    audio_file = self.data_mode_pd_file[index]
    audio_mode = self.data_mode_pd_mode[index]
    audio_label = self.data_mode_pd_label[index]
    assert audio_mode == self.mode_type, "[ERROR:] Something wronge about mode, please check"

    # print('Init Time: {}'.format((time.time() - begin_t) * 1.0))
    # begin_t = time.time() 

    # gen label
    label = self.positive_words_index[audio_label]

    # load data
    case_input_dir = os.path.join(self.input_dir, audio_label)
    if audio_label == SILENCE_LABEL:
      filename = str(label) + '_' + audio_label + '_' + str(index) + '.txt'
    else:
      filename = str(label) + '_' + os.path.basename(os.path.dirname(audio_file)) + '_' + os.path.basename(audio_file).split('.')[0] + '.txt'
    
    f = open(os.path.join(case_input_dir, filename), 'rb')
    data = pickle.load(f)
    f.close()
    
    # print('Load data Time: {}'.format((time.time() - begin_t) * 1.0))

    # To tensor
    data_tensor = torch.from_numpy(data.reshape(1, -1, 40))
    data_tensor = data_tensor.float()
    label_tensor = torch.tensor(label)

    # check tensor
    assert data_tensor.shape[0] == self.input_channel
    assert data_tensor.shape[1] == self.data_size_h
    assert data_tensor.shape[2] == self.data_size_w
    return data_tensor, label_tensor, index
    
    

