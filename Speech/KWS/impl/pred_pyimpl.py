import importlib
# import librosa
import lmdb
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


def load_model(model_folder, epoch, sub_folder_name='checkpoints'):
    """
    :param model_folder:           The folder containing models
    :param epoch:                  The epoch of the model
    :return:
      model_dict:                  The loaded pytorch model
    """
    if epoch < 0:
        last_checkpoint_folder = last_checkpoint(os.path.join(model_folder, sub_folder_name))
        params_file = os.path.join(last_checkpoint_folder, 'parameter.pkl')
    else:
        params_file = os.path.join(model_folder, sub_folder_name, 'chk_{}'.format(epoch), 'parameter.pkl')

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


def kws_load_model(model_folder, gpu_id, epoch, sub_folder_name='checkpoints'):
    """ Load pytorch model for kws
    :param model_folder:           The folder containing pytorch models
    :param gpu_id:                 The ID of the gpu to run model
    :param epoch:                  The epoch of the model
    :param sub_folder_name:        The subpath of the model
    :return:
      model:                       The loaded pytorch model
    """
    assert isinstance(gpu_id, int)

    # switch to specific gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_id)
    assert torch.cuda.is_available(), 'CUDA is not available.'

    model = dict()
    model['prediction'] = load_model(model_folder, epoch, sub_folder_name)
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


def load_preload_audio(audio_file, audio_idx, audio_label, input_dir, refilename=True):
    # load data
    if refilename:
        if audio_label == SILENCE_LABEL:
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

    
def load_lmdb_env(lmdb_path):
    lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    return lmdb_env


def read_audio_lmdb(env, key):
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode())
    audio_data = np.frombuffer(buf, dtype=np.float32)
    return audio_data


def load_background_noise_lmdb(cfg):
    # load noise data
    # init 
    background_data_pd = pd.read_csv(cfg.general.background_data_path)
    background_data_lmdb_path = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date), 'dataset_audio_lmdb', '{}.lmdb'.format(BACKGROUND_NOISE_DIR_NAME))
    background_data = []

    background_data_lmdb_env = lmdb.open(background_data_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    for _, row in background_data_pd.iterrows():
        file_path = row.file
        background_data.append(read_audio_lmdb(background_data_lmdb_env, file_path))
    return background_data


def dataset_add_noise(cfg, data, background_data, bool_silence_label=False):
    assert len(background_data) > 0, "[ERROR:] Something wronge with background data, please check"

    # init 
    background_frequency = cfg.dataset.augmentation.background_frequency
    background_volume = cfg.dataset.augmentation.background_volume

    # add noise
    background_clipped = np.zeros(len(data))
    background_volume_clipped = 0

    if background_frequency > 0:
        background_index = np.random.randint(len(background_data))
        background_samples = background_data[background_index]
        assert len(background_samples) >= len(data), \
            "[ERROR:] Background sample is too short! Need more than {} samples but only {} were found".format(len(data), len(background_samples))
        background_offset = np.random.randint(0, len(background_samples) - len(data) - 1)
        background_clipped = background_samples[background_offset:(background_offset + len(data))]
            
    if np.random.uniform(0, 1) < background_frequency or bool_silence_label:
        background_volume_clipped = np.random.uniform(0, background_volume)

    data_min_value = min(data.max(), abs(data.min()))
    background_min_value = min((background_volume * background_clipped).max(), abs((background_volume * background_clipped).min()))
    if background_min_value < data_min_value or bool_silence_label:
        data = background_volume_clipped * background_clipped + data 

    # data clip 
    data = np.clip(data, -1.0, 1.0) 
    return data 


def audio_preprocess(cfg, data):
    """ 
    :param cfg:                   The config 
    :param data:                  The input data
    :return:
      audio_data:                 The model input, audio features
    """
    # init 
    audio_preprocess_type = cfg.dataset.preprocess
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    window_size_ms = cfg.dataset.window_size_ms
    window_stride_ms = cfg.dataset.window_stride_ms
    feature_bin_count = cfg.dataset.feature_bin_count
    nfilt = cfg.dataset.nfilt
    desired_samples = int(sample_rate * clip_duration_ms / 1000)

    # init audio_processor
    audio_processor = AudioPreprocessor(sr=sample_rate, 
                                        n_mels=feature_bin_count, 
                                        nfilt=nfilt,
                                        winlen=window_size_ms / 1000, 
                                        winstep=window_stride_ms / 1000,
                                        data_length=clip_duration_ms / 1000)
    # check
    assert audio_preprocess_type in [
        "mfcc", "pcen", "fbank", "fbank_cpu"], "[ERROR:] Audio preprocess type is wronge, please check"

    # preprocess
    if audio_preprocess_type == "mfcc":
        audio_data = audio_processor.compute_mfccs(data)
    elif audio_preprocess_type == "pcen":
        audio_data = audio_processor.compute_pcen(data)
    elif audio_preprocess_type == "fbank":
        audio_data = audio_processor.compute_fbanks(data)
    elif audio_preprocess_type == "fbank_cpu":
        audio_data = audio_processor.compute_fbanks_cpu(data)
    return audio_data
    
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
    data = audio_preprocess(cfg, data)

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