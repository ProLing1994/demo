import builtins
import importlib
import numpy as np
import os
import pandas as pd
import sys
import shutil
import torch

sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.file_tools import load_module_from_disk
from common.common.utils.python.train_tools import EpochConcateSampler

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/SV')
from dataset.sv_dataset_preload_audio_lmdb import SpeakerVerificationDataset, SpeakerVerificationDataLoader
from utils.lmdb_tools import *

def load_cfg_file(config_file):
    """
    :param config_file:  configure file path
    :return: cfg:        configuration file module
    """
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)
    cfg = load_module_from_disk(config_file)
    cfg = cfg.cfg

    return cfg

 
def setup_workshop(cfg):
    """
    :param cfg:  training configure file
    :return:     None
    """
    if cfg.general.resume_epoch < 0 and os.path.isdir(cfg.general.save_dir):
        sys.stdout.write(
            "Found non-empty save dir: {} \nType 'yes' to delete, 'no' to continue: ".format(cfg.general.save_dir))
        choice = builtins.input().lower()
        if choice == 'yes':
            shutil.rmtree(cfg.general.save_dir)
        elif choice == 'no':
            pass
        else:
            raise ValueError("Please type either 'yes' or 'no'!")


def init_torch_and_numpy(cfg, local_rank=0):
    """ enable cudnn and control randomness during training, 设置随机种子，以使得结果是确定的
    :param cfg:         configuration file
    :param local_rank:  the device index
    :return:     None
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.general.gpu_ids
    assert torch.cuda.is_available(), \
        'CUDA is not available! Please check nvidia driver!'
    os.environ['PYTHONHASHSEED'] = str(cfg.debug.seed)
    np.random.seed(cfg.debug.seed)
    torch.manual_seed(cfg.debug.seed)
    torch.cuda.manual_seed(cfg.debug.seed)
    torch.backends.cudnn.benchmark = True

    if cfg.general.data_parallel_mode == 0:
        pass
    elif cfg.general.data_parallel_mode == 1:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
    else:
        raise Exception("[ERROR:] Unknow data parallel mode, please check!")


def import_network(cfg, model_name, class_name):
    """ import network
    :param cfg:
    :return:
    """
    net_module = importlib.import_module('network.' + model_name)
    net = net_module.__getattribute__(class_name)(cfg)
    gpu_ids = list(range(cfg.general.num_gpus))

    if cfg.general.data_parallel_mode == 0:
        net = torch.nn.parallel.DataParallel(net, device_ids=gpu_ids)
        net = net.cuda()
    elif cfg.general.data_parallel_mode == 1:
        net = net.cuda()
        net = torch.nn.parallel.DistributedDataParallel(net)
    else:
        raise Exception("[ERROR:] Unknow data parallel mode, please check!")
    return net


def set_optimizer(cfg, net):
    """
    :param cfg:   training configure file
    :param net:   pytorch network
    :return:
    """
    if cfg.train.optimizer == 'SGD':
        opt = torch.optim.SGD(net.parameters(),
                              lr=cfg.train.lr,
                              momentum=cfg.train.momentum,
                              weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == 'Adam':
        opt = torch.optim.Adam(net.parameters(),
                               lr=cfg.train.lr,
                               betas=cfg.train.betas,
                               weight_decay=cfg.train.weight_decay)
    else:
        raise ValueError('Unknown loss optimizer')

    return opt


def set_scheduler(cfg, optimizer):
    """
    :param cfg:   training configure file
    :param optimizer:   pytorch optimizer
    :return:
    """
    scheduler = None
    if cfg.train.optimizer == 'SGD':
        if cfg.train.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                        step_size=cfg.train.lr_step_size, 
                                                        gamma=cfg.train.lr_gamma)
        elif cfg.train.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, 
                                                        T_0=cfg.train.T_0,
                                                        T_mult=cfg.train.T_mult)
        else:
            raise ValueError('Unknown loss scheduler')
    return scheduler


def load_checkpoint(net, epoch_num, net_dir, optimizer=None, sub_folder_name='checkpoints'):
    """
    load network parameters from directory
    :param epoch_num: the epoch idx of model to load
    :param net: the network object
    :param net_dir: the network directory
    :return: loaded epoch index, loaded batch index
    """
    chk_file = os.path.join(net_dir, sub_folder_name,
                            'chk_{}'.format(epoch_num), 'parameter.pkl')
    if not os.path.isfile(chk_file):
        raise ValueError('checkpoint file not found: {}'.format(chk_file))

    state = torch.load(chk_file)
    net.load_state_dict(state['state_dict'])

    if optimizer:
        optimizer.load_state_dict(state['optimizer'])

    return state['epoch'], state['batch']


def generate_dataset(cfg, mode):
    """
    :param cfg:            config contain data set information
    :param mode:           Which partition to use, must be 'training', 'validation', or 'testing'.
    :return:               data loader, length of data set
    """
    assert mode in ['training', 'testing',
                    'validation'], "[ERROR:] Unknow mode: {}".format(mode)

    # load data_pd
    for dataset_idx in range(len(cfg.general.TISV_dataset_list)):
        dataset_name = cfg.general.TISV_dataset_list[dataset_idx]
        csv_path = os.path.join(cfg.general.data_dir, dataset_name + '.csv')
    
        data_pd_temp = pd.read_csv(csv_path)
        if dataset_idx == 0:
            data_pd = data_pd_temp
        else:
            data_pd = pd.concat([data_pd, data_pd_temp])
    data_pd = data_pd[data_pd["mode"] == mode]

    dataset = SpeakerVerificationDataset(cfg, data_pd)
    sampler = EpochConcateSampler(dataset, cfg.train.num_epochs - (cfg.general.resume_epoch if cfg.general.resume_epoch != -1 else 0))
    dataloader = SpeakerVerificationDataLoader(
                    cfg,
                    dataset,
                    cfg.train.speakers_per_batch,
                    cfg.train.utterances_per_speaker,
                    num_workers=cfg.train.num_threads,
                    sampler=sampler)
    return dataloader, len(dataset)