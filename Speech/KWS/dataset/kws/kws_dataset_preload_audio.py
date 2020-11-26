import librosa
import multiprocessing 
import numpy as np
import os
import sys
import pandas as pd 
import pcen
import pickle
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
    self.label_index = load_label_index(cfg.dataset.label.positive_label, cfg.dataset.label.negative_label)

    # load data 
    data_pd = pd.read_csv(cfg.general.data_csv_path)
    background_data_pd = pd.read_csv(cfg.general.background_data_path)
  
    self.mode_type = mode
    self.data_pd = data_pd[data_pd['mode'] == mode]
    self.input_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date), 'dataset_audio', mode)
    self.data_file_list = self.data_pd['file'].tolist()
    self.data_mode_list = self.data_pd['mode'].tolist()
    self.data_label_list = self.data_pd['label'].tolist()

    self.positive_label = cfg.dataset.label.positive_label[0]
    self.sample_rate = cfg.dataset.sample_rate
    self.clip_duration_ms = cfg.dataset.clip_duration_ms
    self.window_size_ms = cfg.dataset.window_size_ms
    self.window_stride_ms = cfg.dataset.window_stride_ms
    self.feature_bin_count = cfg.dataset.feature_bin_count

    self.input_channel = cfg.dataset.input_channel
    self.data_size_h = cfg.dataset.data_size[1]
    self.data_size_w = cfg.dataset.data_size[0]

    self.augmentation_on = cfg.dataset.augmentation.on and augmentation_on
    self.augmentation_speed_volume_on = cfg.dataset.augmentation.speed_volume_on
    self.background_frequency = cfg.dataset.augmentation.background_frequency
    self.background_volume = cfg.dataset.augmentation.background_volume
    self.time_shift_ms = cfg.dataset.augmentation.time_shift_ms
    self.time_shift_multiple = cfg.dataset.augmentation.time_shift_multiple
    self.possitive_speed_list = cfg.dataset.augmentation.possitive_speed.split(',')
    self.possitive_volume_list = cfg.dataset.augmentation.possitive_volume.split(',')

    self.augmentation_spec_on = cfg.dataset.augmentation.spec_on
    self.F = cfg.dataset.augmentation.F
    self.T = cfg.dataset.augmentation.T
    self.num_masks = cfg.dataset.augmentation.num_masks

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

    self.save_audio_inputs_bool = cfg.debug.save_inputs
    self.save_audio_inputs_dir = cfg.general.save_dir

  def __len__(self):
    """ get the number of images in this dataset """
    return len(self.data_file_list)

  def save_audio(self, data, audio_label, filename):
    out_folder = os.path.join(self.save_audio_inputs_dir, self.mode_type + '_audio', audio_label)

    if not os.path.isdir(out_folder):
      os.makedirs(out_folder)

    filename = filename.split('.')[0] + '.wav'
    librosa.output.write_wav(os.path.join(out_folder, filename), data, sr=self.sample_rate)

  def audio_preprocess(self, data):
    # check 
    assert self.audio_preprocess_type in ["mfcc", "pcen", "fbank"], "[ERROR:] Audio preprocess type is wronge, please check"

    # preprocess
    if self.audio_preprocess_type == "mfcc":
      audio_data = self.audio_processor.compute_mfccs(data)
    elif self.audio_preprocess_type == "pcen":
      audio_data = self.audio_processor.compute_pcen(data)
    elif self.audio_preprocess_type == "fbank":
      audio_data = self.audio_processor.compute_fbanks(data)
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

  def dataset_augmentation_volume_speed(self):
    possitive_speed = random.choice(self.possitive_speed_list)
    possitive_volume = random.choice(self.possitive_volume_list)
    return possitive_speed, possitive_volume

  def dataset_augmentation_waveform(self, data, audio_label):
    # add time_shift
    time_shift_amount = 0

    # Time shift enhancement multiple of negative samples
    time_shift_samples = self.time_shift_samples
    if audio_label == UNKNOWN_WORD_LABEL:
      time_shift_samples *= self.time_shift_multiple

    if time_shift_samples > 0:
      time_shift_amount = np.random.randint(-time_shift_samples, time_shift_samples)
    
    time_shift_left = - min(0, time_shift_amount)
    time_shift_right = max(0, time_shift_amount)
    data = np.pad(data, (time_shift_left, time_shift_right), "constant")
    data = data[:len(data) - time_shift_left] if time_shift_left else data[time_shift_right:]

    # add noise
    data = self.dataset_add_noise(data)
    return data 

  def dataset_augmentation_spectrum(self, audio_data):
    # add SpecAugment 
    audio_data = add_frequence_mask(audio_data, F=self.F, num_masks=self.num_masks, replace_with_zero=True)
    audio_data = add_time_mask(audio_data, T=self.T, num_masks=self.num_masks, replace_with_zero=True)
    return audio_data

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

    # load label idx
    audio_label_idx = self.label_index[audio_label]

    # data augmentation
    possitive_speed = '1.0'
    possitive_volume = '1.0'
    if self.augmentation_on and self.augmentation_speed_volume_on:
      possitive_speed, possitive_volume = self.dataset_augmentation_volume_speed()

    # load data
    if audio_label != self.positive_label or (possitive_speed == '1.0' and possitive_volume == '1.0'):
      input_dir = os.path.join(self.input_dir, audio_label)
      data, filename = load_preload_audio(audio_file, index, audio_label, audio_label_idx, input_dir)
    else:
      input_dir = os.path.join(self.input_dir, audio_label + '_speed_{}_volume_{}'.format("_".join(possitive_speed.split('.')), "_".join(possitive_volume.split('.'))))
      data, filename = load_preload_audio(audio_file, index, audio_label, audio_label_idx, input_dir, refilename=False)

    # print('Load data Time: {}'.format((time.time() - begin_t) * 1.0))
    # begin_t = time.time()

    # alignment data
    # data_length = len(data)
    # data = np.pad(data, (max(0, (self.desired_samples - data_length)//2), 0), "constant")
    # data = np.pad(data, (0, max(0, (self.desired_samples - data_length + 1)//2)), "constant")
    if len(data) < self.desired_samples:
      data_length = len(data)
      data_offset = np.random.randint(0, self.desired_samples - data_length - 1)
      data = np.pad(data, (data_offset, 0), "constant")
      data = np.pad(data, (0, self.desired_samples - data_length - data_offset), "constant")

    if len(data) > self.desired_samples:
      data_offset = np.random.randint(0, len(data) - self.desired_samples - 1)
      data = data[data_offset:(data_offset + self.desired_samples)]

    assert len(data) == self.desired_samples, "[ERROR:] Something wronge about audio length, please check"

    if self.save_audio_inputs_bool:
      self.save_audio(data, audio_label, filename)

    # data augmentation
    if audio_label == SILENCE_LABEL:
      data = self.dataset_add_noise(data, bool_silence_label=True)
    elif self.augmentation_on:
      data = self.dataset_augmentation_waveform(data, audio_label)
    # print('Data augmentation Time: {}'.format((time.time() - begin_t) * 1.0))
    # begin_t = time.time()

    # audio preprocess, get mfcc data
    data = self.audio_preprocess(data)
    # print('Audio preprocess Time: {}'.format((time.time() - begin_t) * 1.0))

    # data augmentation
    if self.augmentation_spec_on:
      data = self.dataset_augmentation_spectrum(data)
    
    # To tensor
    data_tensor = torch.from_numpy(data.reshape(1, -1, 40))
    data_tensor = data_tensor.float()
    label_tensor = torch.tensor(audio_label_idx)

    # check tensor
    assert data_tensor.shape[0] == self.input_channel
    assert data_tensor.shape[1] == self.data_size_h
    assert data_tensor.shape[2] == self.data_size_w
    return data_tensor, label_tensor, index
    
    

