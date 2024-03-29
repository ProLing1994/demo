import numpy as np
import os
import sys
import pandas as pd
import torch

from torch.utils.data import Dataset

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio

from SED.utils.folder_tools import *
from SED.utils.lmdb_tools import *
from SED.dataset.dataset_helper import *

class SpeechDataset(Dataset):
    """
    Training dataset for Key word spotting
    """

    def __init__(self, cfg, mode, augmentation_on=True):
        # init
        super().__init__()

        # load data pandas
        data_pd = pd.read_csv(cfg.general.data_csv_path, encoding='utf_8_sig')

        self.mode_type = mode
        self.data_pd = data_pd[data_pd['mode'] == self.mode_type]
        self.data_file_list = self.data_pd['file'].tolist()
        self.data_mode_list = self.data_pd['mode'].tolist()
        self.data_label_list = self.data_pd['label'].tolist()
        self.audios_num = len(self.data_label_list)
        self.classes_num = cfg.dataset.label.num_classes 
        self.audio_label_type = cfg.dataset.label.type 

        # lmdb
        self.lmdb_path = os.path.join(os.path.dirname(cfg.general.data_csv_path), 'dataset_audio_lmdb', '{}.lmdb'.format(mode))
        self.lmdb_env = load_lmdb_env(self.lmdb_path)

        self.sample_rate = cfg.dataset.sample_rate
        self.clip_duration_ms = cfg.dataset.clip_duration_ms
        self.window_size_ms = cfg.dataset.window_size_ms
        self.window_stride_ms = cfg.dataset.window_stride_ms
        self.feature_bin_count = cfg.dataset.feature_bin_count
        self.nfilt = cfg.dataset.nfilt

        self.input_channel = cfg.dataset.input_channel
        self.data_size_h = cfg.dataset.data_size[1]
        self.data_size_w = cfg.dataset.data_size[0]

        self.augmentation_on = cfg.dataset.augmentation.on and augmentation_on

        self.augmentation_background_noise_on = cfg.dataset.augmentation.background_noise_on
        self.background_frequency = cfg.dataset.augmentation.background_frequency
        self.background_volume = cfg.dataset.augmentation.background_volume

        self.augmentation_time_shift_on = cfg.dataset.augmentation.time_shift_on
        self.time_shift_ms = cfg.dataset.augmentation.time_shift_ms

        self.augmentation_speed_volume_on = cfg.dataset.augmentation.speed_volume_on
        self.speed_list = cfg.dataset.augmentation.speed
        self.volume_list = cfg.dataset.augmentation.volume

        self.augmentation_pitch_on = cfg.dataset.augmentation.pitch_on
        self.pitch_list = cfg.dataset.augmentation.pitch 

        self.augmentation_spec_on = cfg.dataset.augmentation.spec_on
        self.F = cfg.dataset.augmentation.F
        self.T = cfg.dataset.augmentation.T
        self.num_masks = cfg.dataset.augmentation.num_masks

        self.desired_samples = int(self.sample_rate * self.clip_duration_ms / 1000)
        self.time_shift_samples = int(self.sample_rate * self.time_shift_ms / 1000)

        self.audio_preprocess_type = cfg.dataset.preprocess
        self.audio_processor = AudioPreprocessor(sr=self.sample_rate,
                                                 n_mels=self.feature_bin_count,
                                                 nfilt=self.nfilt,
                                                 winlen=self.window_size_ms / 1000, 
                                                 winstep=self.window_stride_ms / 1000)

        self.save_audio_bool = cfg.debug.save_inputs
        self.save_audio_dir = cfg.general.save_dir

    def __len__(self):
        """ get the number of images in this dataset """
        return len(self.data_file_list)

    def samples_num_per_class(self):
        assert self.audio_label_type in ["multi_class", "multi_label"], \
            "[ERROR:] Audio label type is wronge, please check"

        if self.audio_label_type == "multi_class":
            # init
            samples_num_per_class = []
            
            data_label_np = np.array(self.data_label_list)
            for idx in range(self.classes_num):
                samples_num_per_class.append(len(data_label_np[data_label_np == idx]))
            return samples_num_per_class
        elif self.audio_label_type == "multi_label":
            targets = np.zeros((self.audios_num, self.classes_num))

            # update targets
            for audio_idx in range(self.audios_num):
                audio_label_id_list = str(self.data_label_list[audio_idx]).split('/')
                for label_id_idx in range(len(audio_label_id_list)):
                    targets[audio_idx][int(audio_label_id_list[label_id_idx])] = 1.0
            return np.sum(targets, axis=0)

    def indexes_per_class(self):
        assert self.audio_label_type in ["multi_class", "multi_label"], \
            "[ERROR:] Audio label type is wronge, please check"
        
        # init
        indexes_per_class = []

        if self.audio_label_type == "multi_class":
            data_label_np = np.array(self.data_label_list)
            for idx in range(self.classes_num):
                indexes_per_class.append(np.where(data_label_np == idx)[0])
        elif self.audio_label_type == "multi_label":
            targets = np.zeros((self.audios_num, self.classes_num))

            # update targets
            for audio_idx in range(self.audios_num):
                audio_label_id_list = str(self.data_label_list[audio_idx]).split('/')
                for label_id_idx in range(len(audio_label_id_list)):
                    targets[audio_idx][int(audio_label_id_list[label_id_idx])] = 1.0

            for k in range(self.classes_num):
                indexes_per_class.append(np.where(targets[:, k] == 1)[0])
        return indexes_per_class
        
    def save_audio(self, data, audio_label, audio_file):
        out_folder = os.path.join(self.save_audio_dir, self.mode_type + '_audio', str("_".join(str(audio_label).split('/'))))
        
        try:
            create_folder(out_folder)
        except:
            pass
        filename = os.path.basename(audio_file)
        audio.save_wav(data.copy(), os.path.join(out_folder, filename), self.sample_rate)

    def audio_preprocess(self, data):
        # check
        assert self.audio_preprocess_type in [
            "mfcc", "pcen", "fbank", "fbank_cpu"], "[ERROR:] Audio preprocess type is wronge, please check"

        # preprocess
        if self.audio_preprocess_type == "mfcc":
            audio_data = self.audio_processor.compute_mfccs(data)
        elif self.audio_preprocess_type == "pcen":
            audio_data = self.audio_processor.compute_pcen(data)
        elif self.audio_preprocess_type == "fbank":
            audio_data = self.audio_processor.compute_fbanks(data)
        elif self.audio_preprocess_type == "fbank_cpu":
            audio_data = self.audio_processor.compute_fbanks_cpu(data)
        return audio_data

    def dataset_alignment(self, data):
        # alignment data
        if len(data) < self.desired_samples:
            data_length = len(data)
            data_offset = np.random.randint(0, self.desired_samples - data_length)
            data = np.pad(data, (data_offset, 0), "constant")
            data = np.pad(data, (0, self.desired_samples - data_length - data_offset), "constant")

        if len(data) > self.desired_samples:
            data_offset = np.random.randint(0, len(data) - self.desired_samples)
            data = data[data_offset:(data_offset + self.desired_samples)]

        assert len(data) == self.desired_samples, "[ERROR:] Something wronge with audio length, please check"
        return data

    def dataset_add_noise(self, data, bool_silence_label=False):
        if not self.background_frequency > 0:
            return data
        
        assert len(self.background_data) > 0, "[ERROR:] Something wronge with background data, please check"

        # init
        background_clipped = np.zeros(self.desired_samples)
        background_volume = 0

        background_idx = np.random.randint(len(self.background_data))
        background_sample = self.background_data[background_idx]
        assert len(background_sample) >= self.desired_samples, \
            "[ERROR:] Background sample is too short! Need more than {} samples but only {} were found".format(
                self.desired_samples, len(background_sample))
        background_offset = np.random.randint(0, len(background_sample) - self.desired_samples - 1)
        background_clipped = background_sample[background_offset:(background_offset + self.desired_samples)]

        if np.random.uniform(0, 1) < self.background_frequency or bool_silence_label:
            background_volume = np.random.uniform(0, self.background_volume)

        data_max_value = data.max()
        background_max_value = (background_volume * background_clipped).max() * 0.8
        if background_max_value < data_max_value or bool_silence_label:
            data = background_volume * background_clipped + data

        # data clip
        data = np.clip(data, -1.0, 1.0)
        return data

    def dataset_augmentation_pitch(self, data):
        pitch = np.random.randint(self.pitch_list[0], self.pitch_list[1])  
        
        # 音调调节
        data = librosa.effects.pitch_shift(data, sr=self.sample_rate, n_steps=pitch)
        return data

    def dataset_augmentation_volume_speed(self, data):
        speed = np.random.uniform(self.speed_list[0], self.speed_list[1])  
        volume = np.random.uniform(self.volume_list[0], self.volume_list[1])  
        
        # speed > 1, 加快速度
        # speed < 1, 放慢速度
        data = librosa.effects.time_stretch(data, speed)

        # 音量大小调节
        data = data * volume
        return data

    def dataset_augmentation_waveform(self, data, audio_label):
        # data augmentation
        if self.augmentation_speed_volume_on:
            data = self.dataset_augmentation_volume_speed(data)
        
        # data augmentation
        if self.augmentation_pitch_on:
            data = self.dataset_augmentation_pitch(data)

        # alignment data
        data = self.dataset_alignment(data) 

        # data augmentation
        if self.augmentation_time_shift_on:
            # add time_shift
            time_shift_amount = 0
            # Time shift enhancement multiple of negative samples
            time_shift_samples = self.time_shift_samples
            if time_shift_samples > 0:
                time_shift_amount = np.random.randint(-time_shift_samples, time_shift_samples)
                time_shift_left = -min(0, time_shift_amount)
                time_shift_right = max(0, time_shift_amount)
                data = np.pad(data, (time_shift_left, time_shift_right), "constant")
                data = data[:len(data) - time_shift_left] if time_shift_left else data[time_shift_right:]

        # data augmentation: add noise
        if self.augmentation_background_noise_on:
            data = self.dataset_add_noise(data)
        return data

    def dataset_augmentation_spectrum(self, audio_data):
        # add SpecAugment
        audio_data = add_frequence_mask(audio_data, F=self.F, num_masks=self.num_masks, replace_with_zero=True)
        audio_data = add_time_mask(audio_data, T=self.T, num_masks=self.num_masks, replace_with_zero=True)
        return audio_data

    def gen_multi_class_multi_label(self, audio_label_id):
        # check
        assert self.audio_label_type in ["multi_class", "multi_label"], \
            "[ERROR:] Audio label type is wronge, please check"
        if self.audio_label_type == "multi_class":
            return audio_label_id
        elif self.audio_label_type == "multi_label":
            audio_label = np.zeros(self.classes_num)
            audio_label_id_list = str(audio_label_id).split('/')
            for idx in range(len(audio_label_id_list)):
                audio_label[int(audio_label_id_list[idx])] = 1.0
            return audio_label 

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
        data = read_audio_lmdb(self.lmdb_env, audio_file)
        assert len(data) != 0, "[ERROR:] Something wronge about load data, please check"
        # print('Load data Time: {}'.format((time.time() - begin_t) * 1.0))
        # begin_t = time.time()

        # data augmentation
        if self.augmentation_on:
            data = self.dataset_augmentation_waveform(data, audio_label)
        else:
            data = self.dataset_alignment(data)
        # print('Data augmentation Time: {}'.format((time.time() - begin_t) * 1.0))
        # begin_t = time.time()

        if self.save_audio_bool:
            self.save_audio(data, audio_label, audio_file)

        # audio preprocess, get mfcc data
        data = self.audio_preprocess(data)
        # print('Audio preprocess Time: {}'.format((time.time() - begin_t) * 1.0))

        # data augmentation
        if self.augmentation_on and self.augmentation_spec_on:
            data = self.dataset_augmentation_spectrum(data)

        # data label
        audio_label = self.gen_multi_class_multi_label(audio_label)

        # To tensor
        data_tensor = torch.from_numpy(np.expand_dims(data, axis=0))
        data_tensor = data_tensor.float()
        label_tensor = torch.tensor(audio_label)
        label_tensor = label_tensor.float()

        # check tensor
        assert data_tensor.shape[0] == self.input_channel
        assert data_tensor.shape[1] == self.data_size_h
        assert data_tensor.shape[2] == self.data_size_w

        # print
        # print(audio_file, audio_label)
        return data_tensor, label_tensor, index
