import importlib
import os
import sys
import torch

from torchstat import stat

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.pred_helpers import last_checkpoint

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

    net = net_module.SpeechResModel(num_classes=12, 
                                    image_height=101, 
                                    image_weidth=40)
    # net = net_module.SpeechResModel(num_classes=state['num_classes'], 
    #                               image_height=state['image_height'], 
    #                               image_weidth=state['image_weidth'])
    net = torch.nn.parallel.DataParallel(net)
    net = net.cuda()
    net.load_state_dict(state['state_dict'])
    net.eval()

    model_dict = {'epoch': state['epoch'],
                  'batch': state['batch'],
                  'net': net}
    # model_dict = {'epoch': state['epoch'],
    #               'batch': state['batch'],
    #               'net': net,
    #               'num_classes': state['num_classes'],
    #               'image_height': state['image_height'],
    #               'image_weidth': state['image_weidth']}
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
