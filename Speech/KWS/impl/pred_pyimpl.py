import importlib
import librosa
import os
import pandas as pd
import pickle
import sys
import torch
import torch.nn.functional as F

from torchstat import stat

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.pred_helpers import last_checkpoint
from dataset.kws.dataset_helper import *


def load_model(model_folder, epoch):
    """
    :param model_folder:           The folder containing models
    :param epoch:                  The epoch of the model
    :return:
      model_dict:                  The loaded pytorch model
    """
    if epoch < 0:
        last_checkpoint_folder = last_checkpoint(os.path.join(model_folder,
                                                              'checkpoints'))
        params_file = os.path.join(last_checkpoint_folder, 'parameter.pkl')
    else:
        params_file = os.path.join(model_folder, 'checkpoints', 'chk_{}'.format(
            epoch), 'parameter.pkl')

    if not os.path.isfile(params_file):
        print('{} params file not found.'.format(params_file))
        return None

    # load model parameters
    state = torch.load(params_file)

    # load network structure
    net_name = state['net']
    net_module = importlib.import_module('network.' + net_name)

    # net = net_module.SpeechResModel(num_classes=12, 
    #                                 image_height=101, 
    #                                 image_weidth=40)
    net = net_module.SpeechResModel(num_classes=state['num_classes'], 
                                  image_height=state['image_height'], 
                                  image_weidth=state['image_weidth'])
    net = torch.nn.parallel.DataParallel(net)
    net = net.cuda()
    net.load_state_dict(state['state_dict'])
    net.eval()

    model_dict = {'epoch': state['epoch'],
                  'batch': state['batch'],
                  'net': net}
    return model_dict


def kws_load_model(model_folder, gpu_id, epoch):
    """ Load pytorch model for kws
    :param model_folder:           The folder containing pytorch models
    :param gpu_id:                 The ID of the gpu to run model
    :param epoch:                  The epoch of the model
    :return:
      model:                       The loaded pytorch model
    """
    assert isinstance(gpu_id, int)

    # switch to specific gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_id)
    assert torch.cuda.is_available(), 'CUDA is not available.'

    model = dict()
    model['prediction'] = load_model(model_folder, epoch)
    model['gpu_id'] = gpu_id

    # switch back to the default gpu
    del os.environ['CUDA_VISIBLE_DEVICES']

    return model


def load_background_noise(cfg):
    # load noise data
    background_data_pd = pd.read_csv(cfg.general.background_data_path)
    input_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date), 'dataset_audio', BACKGROUND_NOISE_DIR_NAME)
    background_data = []
    for _, row in background_data_pd.iterrows():
        filename = os.path.basename(row.file).split('.')[0] + '.txt'
        f = open(os.path.join(input_dir, filename), 'rb')
        background_data.append(pickle.load(f))
        f.close()
    return background_data


def dataset_add_noise(cfg, data, background_data, bool_silence_label=False):
    # init 
    background_frequency = cfg.dataset.augmentation.background_frequency
    background_volume = cfg.dataset.augmentation.background_volume

    # add noise
    background_clipped = np.zeros(len(data))
    background_volume_clipped = 0

    if len(background_data) > 0 and background_frequency > 0:
        background_index = np.random.randint(len(background_data))
        background_samples = background_data[background_index]
        assert len(background_samples) >= len(data), \
            "[ERROR:] Background sample is too short! Need more than {} samples but only {} were found".format(len(data), len(background_samples))
        background_offset = np.random.randint(
            0, len(background_samples) - len(data) - 1)
        background_clipped = background_samples[background_offset:(
            background_offset + len(data))]
            
        if np.random.uniform(0, 1) < background_frequency or bool_silence_label:
            background_volume_clipped = np.random.uniform(0, background_volume)

    data = background_volume_clipped * background_clipped + data 

    # data clip 
    data = np.clip(data, -1.0, 1.0) 
    return data 


def audio_preprocess(cfg, data):
    # init 
    audio_preprocess_type = cfg.dataset.preprocess
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    window_size_ms = cfg.dataset.window_size_ms
    window_stride_ms = cfg.dataset.window_stride_ms
    feature_bin_count = cfg.dataset.feature_bin_count
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    # init audio_processor
    audio_processor = AudioPreprocessor(sr=sample_rate, 
                                        n_dct_filters=feature_bin_count, 
                                        n_fft=window_size_samples, 
                                        hop_length=window_stride_samples)

    # check 
    assert audio_preprocess_type in ["mfcc", "pcen", "fbank"], "[ERROR:] Audio preprocess type is wronge, please check"

    # preprocess
    if audio_preprocess_type == "mfcc":
      audio_data = audio_processor.compute_mfccs(data)
    elif audio_preprocess_type == "pcen":
      audio_data = audio_processor.compute_pcen(data)
    elif audio_preprocess_type == "fbank":
      audio_data = audio_processor.compute_fbanks(data)
    return audio_data 


def model_predict(cfg, model, data):
    # init 
    input_channel = cfg.dataset.input_channel
    data_size_h = cfg.dataset.data_size[1]
    data_size_w = cfg.dataset.data_size[0]

    # audio preprocess, load mfcc data
    data = audio_preprocess(cfg, data)

    # to tensor
    data_tensor = torch.from_numpy(data.reshape(1, -1, 40))
    data_tensor = data_tensor.float()

    # check tensor
    assert data_tensor.shape[0] == input_channel
    assert data_tensor.shape[1] == data_size_h
    assert data_tensor.shape[2] == data_size_w
    
    # infer
    data_tensor = data_tensor.cuda()
    score = model(data_tensor.unsqueeze(0))
    score = F.softmax(score, dim=1)
    score = score.cpu().data.numpy()
    return score