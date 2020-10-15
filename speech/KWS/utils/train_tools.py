import builtins
import importlib
import numpy as np
import os
import sys
import shutil
import torch
import torch.nn as nn

sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.file_tools import load_module_from_disk
from common.common.utils.python.train_tools  import EpochConcateSampler
sys.path.insert(0, '/home/huanyuan/code/demo/speech/KWS')
from dataset.kws.kws_dataset import SpeechDataset

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
      "Found non-empty save dir.\nType 'yes' to delete, 'no' to continue: ")
    choice = builtins.input().lower()
    if choice == 'yes':
      shutil.rmtree(cfg.general.save_dir)
    elif choice == 'no':
      pass
    else:
      raise ValueError("Please type either 'yes' or 'no'!")


def init_torch_and_numpy(cfg):
  """ enable cudnn and control randomness during training
  :param cfg:  configuration file
  :return:     None
  """
  assert torch.cuda.is_available(), \
    'CUDA is not available! Please check nvidia driver!'
  torch.backends.cudnn.benchmark = True
  np.random.seed(cfg.debug.seed)
  torch.manual_seed(cfg.debug.seed)
  torch.cuda.manual_seed(cfg.debug.seed)


def import_network(cfg):
  """ import network
  :param cfg:
  :return:
  """
  assert torch.cuda.is_available(), \
    'CUDA is not available! Please check nvidia driver!'

  net_module = importlib.import_module('network.' + cfg.net.name)
  net = net_module.SpeechResModel(num_classes=cfg.general.num_classes, 
                                  image_height=cfg.dataset.data_size[1], 
                                  image_weidth=cfg.dataset.data_size[0])
  net_module.parameters_init(net)
  gpu_ids = list(range(cfg.general.num_gpus))
  net = torch.nn.parallel.DataParallel(net, device_ids=gpu_ids)
  net = net.cuda()
  # net.train()
  return net


def define_loss_function(cfg):
  """ setup loss function
  :param cfg:
  :return:
  """
  if cfg.loss.name == 'softmax':
    loss_func = nn.CrossEntropyLoss()
  else:
    raise ValueError('Unsupported loss function.')

  return loss_func


def load_checkpoint(epoch_idx, net, save_dir):
  """
  load network parameters from directory
  :param epoch_idx: the epoch idx of model to load
  :param net: the network object
  :param save_dir: the save directory
  :return: loaded epoch index, loaded batch index
  """
  chk_file = os.path.join(save_dir,
                          'checkpoints', 'chk_{}'.format(epoch_idx), 'params.pth')
  if not os.path.isfile(chk_file):
    raise ValueError('checkpoint file not found: {}'.format(chk_file))

  state = load_pytorch_model(chk_file)
  net.load_state_dict(state['state_dict'])

  return state['epoch'], state['batch']


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

def generate_training_dataset(cfg):
  """
  :param cfg:            config contain data set information
  :return:               data loader, length of data set
  """
  data_set = SpeechDataset(mode='train', config=cfg)

  sampler = EpochConcateSampler(data_set, cfg.train.num_epochs - \
                                cfg.general.resume_epoch)

  data_loader = torch.utils.data.DataLoader(data_set,
                                            sampler=sampler,
                                            batch_size=cfg.train.batch_size,
                                            num_workers=cfg.train.num_threads,
                                            pin_memory=True,
                                            worker_init_fn=worker_init)
  return data_loader, len(data_set)