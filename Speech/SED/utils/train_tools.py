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

from tqdm import tqdm

sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech/SED')
# sys.path.insert(0, '/home/huanyuan/code/demo/Speech/SED')
from utils.optimizer_tools import *
from utils.sampler_tools import *
from dataset.dataset_preload_lmdb import SpeechDataset

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo')
# sys.path.insert(0, '/yuanhuan/code/demo')
sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.file_tools import load_module_from_disk
from common.common.utils.python.plotly_tools import plot_loss4d, plot_loss2d, plot_loss

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
        sys.stdout.write("Found non-empty save dir: {} \nType 'yes' to delete, 'no' to continue: ".format(cfg.general.save_dir))
        choice = builtins.input().lower()
        if choice == 'yes':
            shutil.rmtree(cfg.general.save_dir)
        elif choice == 'no':
            pass
        else:
            raise ValueError("Please type either 'yes' or 'no'!")
    
def init_torch_and_numpy(cfg, local_rank=0):
    """ enable cudnn and control randomness during training
    :param cfg:         configuration file
    :param local_rank:  the device index
    :return:     None
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.general.gpu_ids
    assert torch.cuda.is_available(), \
        'CUDA is not available! Please check nvidia driver!'
    torch.backends.cudnn.benchmark = True
    np.random.seed(cfg.debug.seed)
    torch.manual_seed(cfg.debug.seed)
    torch.cuda.manual_seed(cfg.debug.seed)

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

def generate_dataset(cfg, mode):
    """
    :param cfg:            config contain data set information
    :param mode:           Which partition to use, must be 'training', 'validation', or 'testing'.
    :return:               data loader, length of data set
    """
    assert mode in ['training', 'testing', 'validation'], \
        "[ERROR:] Unknow mode: {}".format(mode)

    data_set = SpeechDataset(cfg=cfg, mode=mode)
    data_sampler = TrainSampler(data_set, cfg)
    data_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                                batch_sampler=data_sampler, 
                                                num_workers=cfg.train.num_threads, 
                                                pin_memory=True)
    return data_loader, data_sampler, len(data_set)

def generate_test_dataset(cfg, mode='testing', augmentation_on=False):
    """
    :param cfg:            config contain data set information
    :param mode:           Which partition to use, must be 'training', 'validation', or 'testing'.
    :return:               data loader, length of data set
    """
    assert mode in ['training', 'testing',
                    'validation'], "[ERROR:] Unknow mode: {}".format(mode)

    data_set = SpeechDataset(cfg=cfg, mode=mode, augmentation_on=augmentation_on)
    data_sampler = EvaluateSampler(data_set, cfg)
    data_loader = torch.utils.data.DataLoader(dataset=data_set, 
                                                batch_sampler=data_sampler, 
                                                num_workers=cfg.train.num_threads, 
                                                pin_memory=False)
    return data_loader
    
def save_checkpoint(net, optimizer, sampler, epoch_idx, iteration, cfg, config_file, output_folder_name='checkpoints'):
    """
    save model and parameters into a checkpoint file (.pth)
    :param net: the network object
    :param optimizer: the optimizer object
    :param sampler: the sampler object
    :param epoch_idx: the epoch index
    :param iteration: the iteration
    :param cfg: the configuration object
    :param config_file: the configuration file path
    :return: None
    """
    chk_folder = os.path.join(cfg.general.save_dir,
                              output_folder_name, 'chk_{}'.format(epoch_idx))
    if not os.path.isdir(chk_folder):
        os.makedirs(chk_folder)
    filename = os.path.join(chk_folder, 'parameter.pkl')

    state = {'net': cfg.net.model_name,
             'num_classes': cfg.dataset.label.num_classes,
             'image_height': cfg.dataset.data_size[1],
             'image_weidth': cfg.dataset.data_size[0],
             'state_dict': net.state_dict(),
             'optimizer': optimizer.state_dict(),
             'sampler':sampler.state_dict(),
             'iteration':iteration,
             }
    torch.save(state, filename)
    # 用于在单卡和cpu上加载模型
    # torch.save(net.cpu().module.state_dict(), os.path.join(chk_folder, 'net_parameter.pkl'))
    shutil.copy(config_file, os.path.join(chk_folder, 'config.py'))

def load_checkpoint(net, epoch_num, net_dir, optimizer=None, sampler=None, sub_folder_name='checkpoints'):
    """
    load network parameters from directory
    :param net: the network object
    :param epoch_num: the epoch idx of model to load
    :param net_dir: the network directory
    :param optimizer: the optimizer object
    :param sampler: the sampler object
    :return: loaded epoch index, loaded batch index
    """
    chk_file = os.path.join(net_dir, 
                            sub_folder_name,
                            'chk_{}'.format(epoch_num), 
                            'parameter.pkl')
    if not os.path.isfile(chk_file):
        raise ValueError('checkpoint file not found: {}'.format(chk_file))

    state = torch.load(chk_file)
    net.load_state_dict(state['state_dict'])
    iteration = state['iteration']

    if optimizer:
        optimizer.load_state_dict(state['optimizer'])

    if sampler:
        sampler.load_state_dict(state['sampler'])

    return iteration

def plot_tool(cfg, log_file):
    """
    plot loss or accuracy
    :param cfg:                 config contain data set information
    :param log_file:            log_file
    """
    train_loss_file = os.path.join(cfg.general.save_dir, 'train_loss.html')
    train_accuracy_file = os.path.join(cfg.general.save_dir, 'train_accuracy.html')
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

    # mkdir
    try:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    except:
        pass

    # plot spectrogram
    filename = label_idx + '_' + os.path.basename(os.path.dirname(image_name_idx)) + '_' + os.path.basename(image_name_idx).split('.')[0] + '.jpg'
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
