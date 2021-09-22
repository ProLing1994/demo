import builtins
import importlib
import numpy as np
import os
import pandas as pd
import sys
import shutil
import torch

sys.path.insert(0, '/home/huanyuan/code/demo')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo')
from common.common.utils.python.train_tools import EpochConcateSampler
from common.common.utils.python.plotly_tools import plot_loss2d, plot_loss

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.utils.lmdb_tools import *
from SV.dataset.sv_dataset_preload_audio_lmdb import SpeakerVerificationDataset, SpeakerVerificationDataLoader


def generate_dataset(cfg, mode):
    """
    :param cfg:            config contain data set information
    :param mode:           Which partition to use, must be 'training', 'validation', or 'testing'.
    :return:               data loader, length of data set
    """
    assert mode in ['training', 'testing',
                    'validation'], "[ERROR:] Unknow mode: {}".format(mode)

    dataset = SpeakerVerificationDataset(cfg, mode)
    sampler = EpochConcateSampler(dataset, cfg.train.num_epochs - (cfg.general.resume_epoch if cfg.general.resume_epoch != -1 else 0))
    dataloader = SpeakerVerificationDataLoader(
                    dataset,
                    cfg.train.speakers_per_batch,
                    num_workers=cfg.train.num_threads,
                    sampler=sampler)
    return dataloader, len(dataset)


def generate_test_dataset(cfg, mode):
    """
    :param cfg:            config contain data set information
    :param mode:           Which partition to use, must be 'training', 'validation', or 'testing'.
    :return:               data loader, length of data set
    """
    assert mode in ['training', 'testing',
                    'validation'], "[ERROR:] Unknow mode: {}".format(mode)

    dataset = SpeakerVerificationDataset(cfg, mode, augmentation_on=False)
    dataloader = SpeakerVerificationDataLoader(
                    dataset,
                    1,
                    pin_memory=False,
                    num_workers=1)
    return dataloader


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
        plot_loss2d(log_file, train_accuracy_file, name=['train_eer', 'eval_eer'],
                    display='Training/Validation Accuracy ({})'.format(cfg.loss.name))
    else:
        plot_loss(log_file, train_loss_file, name='train_loss',
                display='Training Loss ({})'.format(cfg.loss.name))
        plot_loss(log_file, train_accuracy_file, name='train_eer',
                display='Training Accuracy ({})'.format(cfg.loss.name))