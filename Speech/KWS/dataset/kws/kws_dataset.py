import librosa
import multiprocessing 
import numpy as np
import os
import pandas as pd 
import pcen
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
    background_data_pd = pd.read_csv(cfg.general.background_data_path)
  
    self.mode_type = mode
    self.data_pd = data_pd[data_pd['mode'] == mode]
    self.data_file_list = self.data_pd['file'].tolist()
    self.data_mode_list = self.data_pd['mode'].tolist()
    self.data_label_list = self.data_pd['label'].tolist()

    self.sample_rate = cfg.dataset.sample_rate
    self.clip_duration_ms = cfg.dataset.clip_duration_ms
    self.window_size_ms = cfg.dataset.window_size_ms
    self.window_stride_ms = cfg.dataset.window_stride_ms
    self.feature_bin_count = cfg.dataset.feature_bin_count

    self.input_channel = cfg.dataset.input_channel
    self.data_size_h = cfg.dataset.data_size[1]
    self.data_size_w = cfg.dataset.data_size[0]

    self.augmentation_on = cfg.dataset.augmentation.on and augmentation_on
    self.background_frequency = cfg.dataset.augmentation.background_frequency
    self.background_volume = cfg.dataset.augmentation.background_volume
    self.time_shift_ms = cfg.dataset.augmentation.time_shift_ms

    self.desired_samples = int(self.sample_rate * self.clip_duration_ms / 1000)
    self.window_size_samples = int(self.sample_rate * self.window_size_ms / 1000)
    self.window_stride_samples = int(self.sample_rate * self.window_stride_ms / 1000)
    self.time_shift_samples =int(self.sample_rate * self.time_shift_ms / 1000)
    self.background_data = [librosa.core.load(row.file, sr=self.sample_rate)[0] for idx, row in background_data_pd.iterrows()]

    self.audio_preprocess_type = cfg.dataset.preprocess
    self.audio_processor = AudioPreprocessor(sr=self.sample_rate, 
                                            n_dct_filters=self.feature_bin_count, 
                                            n_fft=self.window_size_samples, 
                                            hop_length=self.window_stride_samples)

  def __len__(self):
    """ get the number of images in this dataset """
    return len(self.data_file_list)

  def audio_preprocess(self, data):
    # check 
    assert self.audio_preprocess_type in ["mfcc", "pcen"], "[ERROR:] Audio preprocess type is wronge, please check"

    # preprocess
    if self.audio_preprocess_type == "mfcc":
      audio_data = self.audio_processor.compute_mfccs(data)
    elif self.audio_preprocess_type == "pcen":
      audio_data = self.audio_processor.compute_pcen(data)
    return audio_data 

  def dataset_add_noise(self, data, bool_silence_label=False):
    # add noise
    background_clipped = np.zeros(self.desired_samples)
    background_volume = 0

    if len(self.background_data) > 0 and self.background_frequency > 0:
      background_index = np.random.randint(len(self.background_data))
      background_samples = self.background_data[background_index]
      assert len(background_samples) >= self.desired_samples, "[ERROR:] Background sample is too short! Need more than {} samples but only {} were found".format(self.desired_samples, len(background_samples))
      background_offset = np.random.randint(
          0, len(background_samples) - self.desired_samples - 1)
      background_clipped = background_samples[background_offset:(
          background_offset + self.desired_samples)]
          
      if np.random.uniform(0, 1) < self.background_frequency or bool_silence_label:
        background_volume = np.random.uniform(0, self.background_volume)
    
    data = background_volume * background_clipped + data 
    
    # data clip 
    data = np.clip(data, -1.0, 1.0) 
    return data 

  def dataset_augmentation(self, data):
    # add time_shift
    time_shift_amount = 0

    if self.time_shift_samples > 0:
      time_shift_amount = np.random.randint(-self.time_shift_samples, self.time_shift_samples)
    
    time_shift_left = - min(0, time_shift_amount)
    time_shift_right = max(0, time_shift_amount)
    data = np.pad(data, (time_shift_left, time_shift_right), "constant")
    data = data[:len(data) - time_shift_left] if time_shift_left else data[time_shift_right:]

    # add noise
    data = self.dataset_add_noise(data)
    return data 

  def __getitem__(self, index):
    """ get the item """
    # record time
    # begin_t = time.time() 

    audio_file = self.data_file_list[index]
    audio_mode = self.data_mode_list[index]
    audio_label = self.data_label_list[index]
    assert audio_mode == self.mode_type, "[ERROR:] Something wronge about mode, please check"
    # print('Init Time: {}'.format((time.time() - begin_t) * 1.0))
    # begin_t = time.time() 

    # load data
    if audio_label == SILENCE_LABEL:
      data = np.zeros(self.desired_samples, dtype=np.float32)
    else:
      data = librosa.core.load(audio_file, sr=self.sample_rate)[0]
    # print('Load data Time: {}'.format((time.time() - begin_t) * 1.0))
    # begin_t = time.time()

    # alignment data
    data = np.pad(data, (0, max(0, self.desired_samples - len(data))), "constant")
    assert len(data) == self.desired_samples, "[ERROR:] Something wronge about audio length, please check"

    # data augmentation
    if audio_label == SILENCE_LABEL:
      data = self.dataset_add_noise(data, bool_silence_label=True)
    elif self.mode_type == 'training' and self.augmentation_on:
      data = self.dataset_augmentation(data)
    # print('Data augmentation Time: {}'.format((time.time() - begin_t) * 1.0))
    # begin_t = time.time()

    # audio preprocess, get mfcc data
    data = self.audio_preprocess(data)
    # print('Audio preprocess Time: {}'.format((time.time() - begin_t) * 1.0))

    # To tensor
    data_tensor = torch.from_numpy(data.reshape(1, -1, 40))
    data_tensor = data_tensor.float()
    label = self.label_index[audio_label]
    label_tensor = torch.tensor(label)

    # check tensor
    assert data_tensor.shape[0] == self.input_channel
    assert data_tensor.shape[1] == self.data_size_h
    assert data_tensor.shape[2] == self.data_size_w
    return data_tensor, label_tensor, index
    
    

