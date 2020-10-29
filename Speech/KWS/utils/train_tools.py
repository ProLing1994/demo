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

from tqdm import tqdm

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo')
sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.file_tools import load_module_from_disk
from common.common.utils.python.train_tools  import EpochConcateSampler

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech/KWS')
sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
# from dataset.kws.kws_dataset import SpeechDataset
# from dataset.kws.kws_dataset_preprocess import SpeechDataset
from dataset.kws.kws_dataset_preload_audio import SpeechDataset

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


def init_torch_and_numpy(cfg):
  """ enable cudnn and control randomness during training
  :param cfg:  configuration file
  :return:     None
  """
  os.environ['CUDA_VISIBLE_DEVICES'] = cfg.general.gpu_ids
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
  net = net_module.SpeechResModel(num_classes=cfg.dataset.label.num_classes, 
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

def generate_dataset(cfg, mode):
  """
  :param cfg:            config contain data set information
  :param mode:            Which partition to use, must be 'training', 'validation', or 'testing'.
  :return:               data loader, length of data set
  """
  assert mode in ['training', 'testing', 'validation'], "[ERROR:] Unknow mode: {}".format(mode)

  data_set = SpeechDataset(cfg=cfg, mode=mode)
  sampler = EpochConcateSampler(data_set, cfg.train.num_epochs - (cfg.general.resume_epoch if cfg.general.resume_epoch != -1 else 0))
  data_loader = torch.utils.data.DataLoader(data_set,
                                            sampler=sampler,
                                            batch_size=cfg.train.batch_size,
                                            pin_memory=False,
                                            num_workers=cfg.train.num_threads,
                                            worker_init_fn=worker_init)
  return data_loader, len(data_set)

def generate_test_dataset(cfg, mode = 'validation', augmentation_on=False):
  """
  :param cfg:            config contain data set information
  :param mode:           Which partition to use, must be 'training', 'validation', or 'testing'.
  :return:               data loader, length of data set
  """
  assert mode in ['training', 'testing', 'validation'], "[ERROR:] Unknow mode: {}".format(mode)

  data_set = SpeechDataset(cfg=cfg, mode=mode, augmentation_on=augmentation_on)
  data_loader = torch.utils.data.DataLoader(data_set,
                                            batch_size=1,
                                            pin_memory=False,
                                            num_workers=cfg.train.num_threads)
  return data_loader


def load_checkpoint(epoch_idx, net, save_dir):
  """
  load network parameters from directory
  :param epoch_idx: the epoch idx of model to load
  :param net: the network object
  :param save_dir: the save directory
  :return: loaded epoch index, loaded batch index
  """
  chk_file = os.path.join(save_dir,
                          'checkpoints', 'chk_{}'.format(epoch_idx), 'parameter.pkl')
  if not os.path.isfile(chk_file):
    raise ValueError('checkpoint file not found: {}'.format(chk_file))

  state = torch.load(chk_file)
  net.load_state_dict(state['state_dict'])
  return state['epoch'], state['batch']


def save_checkpoint(net, epoch_idx, batch_idx, cfg, config_file):
  """
  save model and parameters into a checkpoint file (.pth)
  :param net: the network object
  :param epoch_idx: the epoch index
  :param batch_idx: the batch index
  :param cfg: the configuration object
  :param config_file: the configuration file path
  :return: None
  """
  chk_folder = os.path.join(cfg.general.save_dir,
                            'checkpoints', 'chk_{}'.format(epoch_idx))
  if not os.path.isdir(chk_folder):
      os.makedirs(chk_folder)
  filename = os.path.join(chk_folder, 'parameter.pkl')

  state = {'epoch': epoch_idx,
            'batch': batch_idx,
            'net': cfg.net.name,
            'num_classes': cfg.dataset.label.num_classes,
            'image_height': cfg.dataset.data_size[1],
            'image_weidth': cfg.dataset.data_size[0],
            'state_dict': net.state_dict(),
            }
  torch.save(state, filename)
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

  out_folder = os.path.join(cfg.general.save_dir, mode)
  if not os.path.isdir(out_folder):
      os.makedirs(out_folder)

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

  image_idx = images[bth_idx].numpy().reshape((-1, 40))
  label_idx = str(labels[bth_idx].numpy())
  index_idx = int(indexs[bth_idx])

  image_name_idx = str(data_pd['file'].tolist()[index_idx])
  label_name_idx = str(data_pd['label'].tolist()[index_idx])
  output_dir = os.path.join(out_folder, label_name_idx)
  if not os.path.isdir(output_dir):
      try:
        os.makedirs(output_dir)
      except:
        pass

  # plot spectrogram
  if label_idx == '0':
    filename = label_idx + '_' + label_name_idx + '_' + str(index_idx) + '.jpg'
  else:
    filename = label_idx + '_' + os.path.basename(os.path.dirname(image_name_idx)) + '_' + os.path.basename(image_name_idx).split('.')[0] + '.jpg'
  plot_spectrogram(image_idx.T, os.path.join(output_dir, filename))
  print("Save Intermediate Results: {}".format(filename))


def plot_spectrogram(image, output_path):
  fig = plt.figure(figsize=(10, 4))
  heatmap = plt.pcolor(image) 
  fig.colorbar(mappable=heatmap)
  plt.xlabel("Time(s)")
  plt.ylabel("MFCC Coefficients")
  plt.tight_layout()
  plt.savefig(output_path, dpi=300)
  plt.close() 