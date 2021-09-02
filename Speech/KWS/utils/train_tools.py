import builtins
import importlib
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo')
# sys.path.insert(0, '/yuanhuan/code/demo')
sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.train_tools import EpochConcateSampler
from common.common.utils.python.file_tools import load_module_from_disk
from common.common.utils.python.plotly_tools import plot_loss4d, plot_loss2d, plot_loss

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech/KWS')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech/KWS')
sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from dataset.kws.dataset_helper import SILENCE_LABEL
from dataset.kws.kws_dataset_align_preload_audio import SpeechDatasetAlign
from dataset.kws.kws_dataset_preload_audio_lmdb import SpeechDataset

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


def import_network(cfg, model_name, class_name='SpeechResModel'):
    """ import network
    :param cfg:
    :return:
    """
    assert torch.cuda.is_available(), \
        'CUDA is not available! Please check nvidia driver!'

    net_module = importlib.import_module('network.' + model_name)
    net = net_module.__getattribute__(class_name)(num_classes=cfg.dataset.label.num_classes,
                                                    image_height=cfg.dataset.data_size[1],
                                                    image_weidth=cfg.dataset.data_size[0])
    net_module.parameters_init(net)
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


def loss_fn_kd(cfg, original_scores, teacher_scores, loss):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    assert cfg.knowledge_distillation.loss_name == 'kd'
    alpha = cfg.knowledge_distillation.alpha
    T = cfg.knowledge_distillation.temperature
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(original_scores/T, dim=1), F.softmax(teacher_scores/T, dim=1)) * (alpha * T * T) + \
                loss * (1. - alpha)

    return KD_loss

def loss_kl(original_scores, teacher_scores):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KL_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(original_scores, dim=1), F.softmax(teacher_scores, dim=1))
    return KL_loss

def worker_init(worker_idx):
    """
    The worker initialization function takes the worker id (an int in "[0,
    num_workers - 1]") as input and does random seed initialization for each
    worker.
    :param worker_idx: The worker index.
    :return: None.
    """
    MAX_INT = sys.maxsize
    worker_seed = np.random.randint(int(np.sqrt(MAX_INT))) + worker_idx
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)


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

def update_scheduler(cfg, scheduler, epoch_idx):
    """
    :param cfg:   training configure file
    :param scheduler:   pytorch scheduler
    :param epoch_idx:   
    :return:
    """
    if cfg.train.optimizer == 'SGD':
        scheduler.step(epoch_idx)
    else:
        pass

def generate_dataset(cfg, mode, training_mode=0):
    """
    :param cfg:            config contain data set information
    :param mode:           Which partition to use, must be 'training', 'validation', or 'testing'.
    :param training_mode:  the model training mode, must be 0 or 1.
    :return:               data loader, length of data set
    """
    assert mode in ['training', 'testing',
                    'validation'], "[ERROR:] Unknow mode: {}".format(mode)

    if training_mode == 0:
        data_set = SpeechDataset(cfg=cfg, mode=mode)
    elif training_mode == 1:
        data_set = SpeechDatasetAlign(cfg=cfg, mode=mode)
    else:
        raise Exception("[ERROR:] Unknow training mode, please check!")

    sampler = EpochConcateSampler(data_set, cfg.train.num_epochs - (
        cfg.general.resume_epoch if cfg.general.resume_epoch != -1 else 0))
    data_loader = torch.utils.data.DataLoader(data_set,
                                              sampler=sampler,
                                              batch_size=cfg.train.batch_size,
                                              pin_memory=True,
                                              num_workers=cfg.train.num_threads,
                                              worker_init_fn=worker_init)
    return data_loader, len(data_set)


def generate_test_dataset(cfg, mode='validation', augmentation_on=False, training_mode=0):
    """
    :param cfg:            config contain data set information
    :param mode:           Which partition to use, must be 'training', 'validation', or 'testing'.
    :param training_mode:  the model training mode, must be 0 or 1.
    :return:               data loader, length of data set
    """
    assert mode in ['training', 'testing',
                    'validation'], "[ERROR:] Unknow mode: {}".format(mode)

    if training_mode == 0:
        data_set = SpeechDataset(
            cfg=cfg, mode=mode, augmentation_on=augmentation_on)
    elif training_mode == 1:
        data_set = SpeechDatasetAlign(
            cfg=cfg, mode=mode, augmentation_on=augmentation_on)
    else:
        raise Exception("[ERROR:] Unknow training mode, please check!")

    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=1,
                                              pin_memory=False,
                                              num_workers=1)
    return data_loader


def generate_dataset_ddp(cfg, mode, training_mode=0):
    """
    :param cfg:            config contain data set information
    :param mode:           Which partition to use, must be 'training', 'validation', or 'testing'.
    :param training_mode:  the model training mode, must be 0 or 1.
    :return:               data loader, length of data set
    """
    assert mode in ['training', 'testing',
                    'validation'], "[ERROR:] Unknow mode: {}".format(mode)

    if training_mode == 0:
        data_set = SpeechDataset(cfg=cfg, mode=mode)
    elif training_mode == 1:
        data_set = SpeechDatasetAlign(cfg=cfg, mode=mode)
    else:
        raise Exception("[ERROR:] Unknow training mode, please check!")

    train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
    train_loader = torch.utils.data.DataLoader(data_set,
                                                sampler=train_sampler,
                                                batch_size=cfg.train.batch_size,
                                                pin_memory=True,
                                                num_workers=cfg.train.num_threads,
                                                worker_init_fn=worker_init)
    return train_loader, train_sampler, len(data_set)


def load_checkpoint(epoch_num, net, net_dir, optimizer=None, sub_folder_name='checkpoints'):
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


def plot_tool(cfg, log_file):
    """
    plot loss or accuracy
    :param cfg:                 config contain data set information
    :param log_file:            log_file
    """
    train_loss_file = os.path.join(cfg.general.save_dir, 'train_loss.html')
    train_accuracy_file = os.path.join(cfg.general.save_dir, 'train_accuracy.html')
    if cfg.deep_mutual_learning.on:
        # Deep Mutual Learning
        if cfg.general.is_test:
            plot_loss4d(log_file, train_loss_file, name=['model_0_train_loss', 'model_1_train_loss', 'model_0_eval_loss', 'model_1_eval_loss'],
                        display='Training/Validation Loss ({})'.format(cfg.loss.name))
            plot_loss4d(log_file, train_accuracy_file, name=['model_0_train_accuracy', 'model_1_train_accuracy', 'model_0_eval_accuracy', 'model_1_eval_accuracy'],
                        display='Training/Validation Accuracy ({})'.format(cfg.loss.name))
        else:
            plot_loss2d(log_file, train_loss_file, name=['model_0_train_loss', 'model_1_train_loss'],
                        display='Training/Validation Loss ({})'.format(cfg.loss.name))
            plot_loss2d(log_file, train_accuracy_file, name=['model_0_train_accuracy', 'model_1_eval_accuracy'],
                        display='Training/Validation Accuracy ({})'.format(cfg.loss.name))

    else:
        if cfg.general.is_test:
            plot_loss2d(log_file, train_loss_file, name=['train_loss', 'eval_loss'],
                        display='Training/Validation Loss ({})'.format(cfg.loss.name))
            plot_loss2d(log_file, train_accuracy_file, name=['train_accuracy', 'eval_accuracy'],
                        display='Training/Validation Accuracy ({})'.format(cfg.loss.name))
        else:
            plot_loss(log_file, train_loss_file, name='train_loss',
                    display='Training Loss ({})'.format(cfg.loss.name))
            plot_loss(log_file, train_accuracy_file, name='train_accuracy',
                    display='Training Accuracy ({})'.format(cfg.loss.name))


def save_checkpoint(cfg, config_file, net, optimizer, epoch_idx, batch_idx, output_folder_name='checkpoints'):
    """
    save model and parameters into a checkpoint file (.pth)
    :param cfg: the configuration object
    :param config_file: the configuration file path
    :param net: the network object
    :param optimizer: the optimizer object
    :param epoch_idx: the epoch index
    :param batch_idx: the batch index
    :return: None
    """
    chk_folder = os.path.join(cfg.general.save_dir, output_folder_name, 'chk_{}'.format(epoch_idx))
    if not os.path.isdir(chk_folder):
        os.makedirs(chk_folder)
    filename = os.path.join(chk_folder, 'parameter.pkl')

    state = {'epoch': epoch_idx,
             'batch': batch_idx,
             'net': cfg.net.model_name,
             'num_classes': cfg.dataset.label.num_classes,
             'image_height': cfg.dataset.data_size[1],
             'image_weidth': cfg.dataset.data_size[0],
             'state_dict': net.state_dict(),
             'optimizer': optimizer.state_dict(),
             }
    torch.save(state, filename)
    # 用于在单卡和cpu上加载模型
    # torch.save(net.cpu().module.state_dict(), os.path.join(chk_folder, 'net_parameter.pkl'))
    shutil.copy(config_file, os.path.join(chk_folder, 'config.py'))


def save_intermediate_results(cfg, mode, epoch, images, labels, indexs):
    """ save intermediate results to training folder
    :param cfg:          config contain data set information
    :param mode:         Which partition to use, must be 'training', 'validation', or 'testing'.
    :param epoch:        the epoch index
    :param images:       the batch images
    :param labels:       the batch labels
    :param indexs:       the batch index
    :return: None
    """
    if epoch != 0:
        return

    print("Save Intermediate Results")

    out_folder = os.path.join(cfg.general.save_dir, mode + '_jpg')
    if not os.path.isdir(out_folder):
        try:
            os.makedirs(out_folder)
        except:
            pass

    # load csv
    data_pd = pd.read_csv(cfg.general.data_csv_path)
    data_pd = data_pd[data_pd['mode'] == mode]

    # mkdir
    label_name_list = data_pd['label'].tolist()
    for label_name_idx in range(len(label_name_list)):
        output_dir = os.path.join(out_folder, str(
            label_name_list[label_name_idx]))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    in_params = []
    batch_size = images.shape[0]
    for bth_idx in tqdm(range(batch_size)):
        in_args = [labels, images, indexs, data_pd, out_folder, bth_idx]
        in_params.append(in_args)

    p = multiprocessing.Pool(cfg.debug.num_processing)
    out = p.map(multiprocessing_save, in_params)
    p.close()
    p.join()
    print("Save Intermediate Results: Done")


def multiprocessing_save(args):
    """ save intermediate results to training folder with multi process
    """
    labels = args[0]
    images = args[1]
    indexs = args[2]
    data_pd = args[3]
    out_folder = args[4]
    bth_idx = args[5]

    image_idx = images[bth_idx].numpy().reshape((-1, images[bth_idx].numpy().shape[-1]))
    label_idx = str(labels[bth_idx].numpy())
    index_idx = int(indexs[bth_idx])

    image_name_idx = str(data_pd['file'].tolist()[index_idx])
    label_name_idx = str(data_pd['label'].tolist()[index_idx])
    output_dir = os.path.join(out_folder, label_name_idx)

    # plot spectrogram
    if label_name_idx == SILENCE_LABEL:
        filename = label_idx + '_' + label_name_idx + \
            '_' + str(index_idx) + '.jpg'
    else:
        filename = label_idx + '_' + os.path.basename(os.path.dirname(
            image_name_idx)) + '_' + os.path.basename(image_name_idx).split('.')[0] + '.jpg'
    plot_spectrogram(image_idx.T, os.path.join(output_dir, filename))
    print("Save Intermediate Results: {}".format(filename))


def plot_spectrogram(image, output_path):
    fig = plt.figure(figsize=(10, 4))
    heatmap = plt.pcolor(image)
    fig.colorbar(mappable=heatmap)
    plt.axis('off')
    plt.xlabel("Time(s)")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
