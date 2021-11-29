import os
import sys

from torch.utils.data import DataLoader

sys.path.insert(0, '/home/huanyuan/code/demo/common')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/common')
from common.utils.python.train_tools import EpochConcateSampler
from common.utils.python.plotly_tools import plot_loss2d, plot_loss

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/TTS')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech/TTS')
# from dataset.tts.sv2tts_dataset_preload_audio_lmdb import SynthesizerDataset, SynthesizerDataLoader
from dataset.tts.sv2tts_dataset_preload_audio_hdf5 import SynthesizerDataset, SynthesizerCollater


def generate_dataset_lmdb(cfg, mode):
    """
    :param cfg:            config contain data set information
    :param mode:           Which partition to use, must be 'training', 'validation', or 'testing'.
    :return:               data loader, length of data set
    """
    assert mode in ['training', 'testing',
                    'validation'], "[ERROR:] Unknow mode: {}".format(mode)

    dataset = SynthesizerDataset(cfg, mode)
    sampler = EpochConcateSampler(dataset, cfg.train.num_epochs - (cfg.general.resume_epoch_num if cfg.general.resume_epoch_num != -1 else 0))
    dataloader = SynthesizerDataLoader(dataset, sampler, cfg)
    return dataloader, len(dataset)


def generate_dataset_hdf5(cfg, mode):
    """
    :param cfg:            config contain data set information
    :param mode:           Which partition to use, must be 'training', 'validation', or 'testing'.
    :return:               data loader, length of data set
    """
    assert mode in ['training', 'testing',
                    'validation'], "[ERROR:] Unknow mode: {}".format(mode)

    dataset = SynthesizerDataset(cfg, mode)
    sampler = EpochConcateSampler(dataset, cfg.train.num_epochs - (cfg.general.resume_epoch_num if cfg.general.resume_epoch_num != -1 else 0))
    collater = SynthesizerCollater(cfg)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=cfg.train.batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.train.num_threads,
                            collate_fn=collater,
                            pin_memory=True)
    return dataloader, len(dataset)


def plot_tool(cfg, log_file):
    """
    plot loss or accuracy
    :param cfg:                 config contain data set information
    :param log_file:            log_file
    """
    train_loss_file = os.path.join(cfg.general.save_dir, 'train_loss.html')
    if cfg.general.is_test:
        plot_loss2d(log_file, train_loss_file, name=['train_loss', 'eval_loss'],
                    display='Training/Validation Loss ({})'.format(cfg.loss.name))
    else:
        plot_loss(log_file, train_loss_file, name='train_loss',
                display='Training Loss ({})'.format(cfg.loss.name))