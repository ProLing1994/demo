import builtins
import glob
import importlib
import numpy as np
import os
import sys
import shutil
import torch

sys.path.insert(0, '/home/huanyuan/code/demo/common')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/common')
# sys.path.insert(0, '/yuanhuan/code/demo/common')
from common.utils.python.file_tools import load_module_from_disk


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
    torch.set_num_threads(cfg.train.num_threads) 

    if cfg.general.data_parallel_mode == 0:
        pass
    elif cfg.general.data_parallel_mode == 1:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
    elif cfg.general.data_parallel_mode == 2:
        pass
    else:
        raise Exception("[ERROR:] Unknow data parallel mode, please check!")


def import_network(cfg, model_name, class_name):
    """ import network
    :param cfg:
    :return:
    """
    os.sys.path.insert(0, os.path.dirname(model_name))
    net_module = importlib.import_module(os.path.splitext(os.path.basename(model_name))[0])
    os.sys.path.pop(0)
    
    # load model
    net = net_module.__getattribute__(class_name)(cfg)
    
    # init model parameters
    if 'parameters_init' in net_module.__dict__:
        net_module.parameters_init(net)
        
    gpu_ids = list(range(cfg.general.num_gpus))
    if cfg.general.data_parallel_mode == 0:
        net = torch.nn.parallel.DataParallel(net, device_ids=gpu_ids)
        net = net.cuda()
    elif cfg.general.data_parallel_mode == 1:
        net = net.cuda()
        net = torch.nn.parallel.DistributedDataParallel(net)
    elif cfg.general.data_parallel_mode == 2:
        net = net.cuda()
    else:
        raise Exception("[ERROR:] Unknow data parallel mode, please check!")
    return net


_output_ref = None
_replicas_ref = None

def data_parallel_workaround(cfg, model, *input):
    """ data parallel workaround
    单机多卡，在数据加载后进行手动并行（目前仅用于 sv2tts 数据并行方式）
    cfg.general.data_parallel_mode == 2 and cfg.general.num_gpus > 1
    :param cfg:
    :param model:
    :param input:
    :return:
    """
    global _output_ref
    global _replicas_ref
    device_ids = list(range(cfg.general.num_gpus))
    output_device = device_ids[0]
    replicas = torch.nn.parallel.replicate(model, device_ids)
    # input.shape = (num_args, batch, ...)
    inputs = torch.nn.parallel.scatter(input, device_ids)
    # inputs.shape = (num_gpus, num_args, batch/num_gpus, ...)
    replicas = replicas[:len(inputs)]
    outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
    y_hat = torch.nn.parallel.gather(outputs, output_device)
    _output_ref = outputs
    _replicas_ref = replicas
    return y_hat


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


def load_checkpoint(net, epoch_num, net_dir, optimizer=None, 
                    sub_folder_name='checkpoints', 
                    state_name='state_dict',
                    optimizer_name='optimizer'):
    """
    load network parameters from directory
    :param net: the network object
    :param epoch_num: the epoch idx of model to load
    :param net_dir: the network directory
    :return: loaded epoch index, loaded batch index
    """
    if epoch_num < 0:
        epoch_num = last_checkpoint(os.path.join(net_dir, sub_folder_name))

    chk_file = os.path.join(net_dir, sub_folder_name,
                            'chk_{}'.format(epoch_num), 'parameter.pkl')
    if not os.path.isfile(chk_file):
        raise ValueError('checkpoint file not found: {}'.format(chk_file))
    
    # 方案一:
    state = torch.load(chk_file)
    net.load_state_dict(state[state_name])

    # # 方案二: 去除 module. 字段
    # state = torch.load(chk_file)
    # new_pre = {}
    # for k, v in state[state_name].items():
    #     name = k[7:]
    #     new_pre[name] = v
    # net.load_state_dict(new_pre)

    if optimizer:
        optimizer.load_state_dict(state[optimizer_name])

    return state['epoch'], state['batch']


def load_checkpoint_from_path(net, chk_file,
                                state_name='state_dict',
                                finetune_ignore_key_list=None):
    # 方案二: 需要添加 module. 字段
    state = torch.load(chk_file)
    new_pre = {}
    for k, v in state[state_name].items():
        name = 'module.' + k
        if finetune_ignore_key_list:
            if name in finetune_ignore_key_list:
                continue
            else:
                new_pre[name] = v
        else:
            new_pre[name] = v
    
    # net.load_state_dict(new_pre)
    net.load_state_dict(new_pre, strict=False)

    return


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
             'model_name': cfg.net.model_name,
             'class_name': cfg.net.class_name,
             'state_dict': net.state_dict(),
             'optimizer': optimizer.state_dict(),
             }
    torch.save(state, filename)
    # 用于在单卡和cpu上加载模型
    # torch.save(net.cpu().module.state_dict(), os.path.join(chk_folder, 'net_parameter.pkl'))
    shutil.copy(config_file, os.path.join(chk_folder, 'config.py'))


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

    return last_epoch