import os
import glob
import numpy as np
import pandas as pd
import pickle
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio 

from KWS.config.kws import hparams


def load_preload_audio(audio_file, audio_idx, audio_label, input_dir, refilename=True):
    # load data
    if refilename:
        if audio_label == hparams.SILENCE_LABEL:
            filename = audio_label + '_' + str(audio_idx) + '.txt'
        else:
            filename = os.path.basename(os.path.dirname(audio_file)) + '_' + os.path.basename(audio_file).split('.')[0] + '.txt'
    else:
        filename =  os.path.basename(audio_file).split('.')[0] + '.txt'

    file_path_list = glob.glob(os.path.join(input_dir, '*' + filename).encode('utf-8'))
    assert len(file_path_list) == 1, "{} {}".format(len(file_path_list), os.path.join(input_dir, '*' + filename).encode('utf-8'))
    f = open(file_path_list[0], 'rb')
    data = pickle.load(f)
    f.close()
    return data, filename


def load_background_noise(cfg):
    # load noise data
    background_data_pd = pd.read_csv(cfg.general.background_data_path)
    input_dir = os.path.join(os.path.dirname(cfg.general.data_csv_path), 'dataset_audio', hparams.BACKGROUND_NOISE_DIR_NAME)
    background_data = []
    for _, row in background_data_pd.iterrows():
        filename = os.path.basename(row.file).split('.')[0] + '.txt'
        f = open(os.path.join(input_dir, filename), 'rb')
        background_data.append(pickle.load(f))
        f.close()
    return background_data


def model_predict(cfg, model, data):
    """ 
    :param cfg:                   The config 
    :param model:                 The pytorch model
    :param data:                  The input data
    :return:
      score:                      The model prediction results
    """
    # init 
    input_channel = cfg.dataset.input_channel
    data_size_h = cfg.dataset.data_size[1]
    data_size_w = cfg.dataset.data_size[0]

    # audio preprocess, load mfcc data
    data = audio.compute_mel_spectrogram(cfg, data)

    if cfg.dataset.h_alignment == True:
        data = data[:(data.shape[0] // 16) * 16, :]
        data_size_h = (data_size_h // 16) * 16

    # to tensor
    data_tensor = torch.from_numpy(np.expand_dims(data, axis=0))
    data_tensor = data_tensor.float()

    # check tensor
    assert data_tensor.shape[0] == input_channel
    assert data_tensor.shape[1] == data_size_h
    assert data_tensor.shape[2] == data_size_w
    
    # infer
    data_tensor = data_tensor.cuda()
    score = model(data_tensor.unsqueeze(0))
    score = F.softmax(score, dim=1)
    score = score.detach().cpu().data.numpy()
    return score