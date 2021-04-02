import glob
import importlib
import os
import torch
import torch.nn.functional as F

from wakeup.dataset_helper import AudioPreprocessor


def last_checkpoint(chk_root):
    """
    find the directory of last check point
    :param chk_root: the check point root directory, which may contain multiple
     checkpoints
    :return: the last check point directory
    """

    last_epoch = -1
    chk_folders = os.path.join(chk_root, 'chk_*')
    for folder in glob.glob(chk_folders):
        folder_name = os.path.basename(folder)
        tokens = folder_name.split('_')
        epoch = int(tokens[-1])
        if epoch > last_epoch:
            last_epoch = epoch

    if last_epoch == -1:
        raise OSError('No checkpoint folder found!')

    return os.path.join(chk_root, 'chk_{}'.format(last_epoch))


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
    net_module = importlib.import_module('wakeup.network.' + net_name)

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
    desired_samples = int(sample_rate * clip_duration_ms / 1000)

    # init audio_processor
    audio_processor = AudioPreprocessor(sr=sample_rate, 
                                        n_mels=feature_bin_count, 
                                        winlen=self.window_size_ms / 1000, 
                                        winstep=self.window_stride_ms / 1000)

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