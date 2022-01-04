import os
import sys

from torch.utils.data import DataLoader

sys.path.insert(0, '/home/huanyuan/code/demo/common')
# sys.path.insert(0, '/yuanhuan/code/demo/common')
from common.utils.python.train_tools import EpochConcateSampler
from common.utils.python.plotly_tools import plot_loss4d, plot_loss2d, plot_loss

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech')
from Basic.utils.train_tools import *

from VOCODER.dataset.vocoder.vocoder_dataset_preload_audio_lmdb import VocoderDataset, VocoderDataLoader
# from VOCODER.dataset.vocoder.vocoder_wavegan_dataset_preload_audio_lmdb import VocoderWaveGanDataset, VocoderWaveGanDataLoader
from VOCODER.dataset.vocoder.vocoder_wavegan_dataset_preload_audio_hdf5 import VocoderWaveGanDataset, VocoderCollater
from VOCODER.dataset.vocoder.vocoder_wavegan_vc_dataset_preload_audio_hdf5 import VocoderWaveGanVcDataset, VocoderVcCollater


def generate_dataset(cfg, mode):
    """
    :param cfg:            config contain data set information
    :param mode:           Which partition to use, must be 'training', 'validation', or 'testing'.
    :return:               data loader, length of data set
    """
    assert mode in ['training', 'testing', 'validation'], "[ERROR:] Unknow mode: {}".format(mode)

    dataset = VocoderDataset(cfg, mode)
    sampler = EpochConcateSampler(dataset, cfg.train.num_epochs - (cfg.general.resume_epoch_num if cfg.general.resume_epoch_num != -1 else 0))
    dataloader = VocoderDataLoader(dataset, sampler, cfg)
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


def generate_dataset_wavegan(cfg, mode):
    assert mode in ['training', 'testing', 'validation'], "[ERROR:] Unknow mode: {}".format(mode)
    
    dataset = VocoderWaveGanDataset(cfg, mode)
    sampler = EpochConcateSampler(dataset, cfg.train.num_epochs - (cfg.general.resume_epoch_num if cfg.general.resume_epoch_num != -1 else 0))
    collater = VocoderCollater(cfg)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=cfg.train.batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.train.num_threads,
                            collate_fn=collater,
                            pin_memory=True)
    return dataloader, len(dataset)


def generate_dataset_wavegan_vc(cfg, mode):
    assert mode in ['training', 'testing', 'validation'], "[ERROR:] Unknow mode: {}".format(mode)
    
    dataset = VocoderWaveGanVcDataset(cfg, mode)
    sampler = EpochConcateSampler(dataset, cfg.train.num_epochs - (cfg.general.resume_epoch_num if cfg.general.resume_epoch_num != -1 else 0))
    collater = VocoderVcCollater(cfg)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=cfg.train.batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.train.num_threads,
                            collate_fn=collater,
                            pin_memory=True)
    return dataloader, len(dataset)


def plot_tool_wavegan(cfg, log_file):
    """
    plot loss or accuracy
    :param cfg:                 config contain data set information
    :param log_file:            log_file
    """
    train_loss_file = os.path.join(cfg.general.save_dir, 'train_loss.html')
    if cfg.general.is_test:
        plot_loss4d(log_file, train_loss_file, name=['train/generator_loss', 'train/discriminator_loss', 'eval/generator_loss', 'eval/discriminator_loss'],
                    display='Training/Validation Loss')
    else:
        plot_loss2d(log_file, train_loss_file, name=['train/generator_loss', 'train/discriminator_loss'],
                display='Training Loss')


def load_checkpoint_wavegan(checkpoint_path, model, optimizer=None, scheduler=None):
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    model["generator"].load_state_dict(state_dict["model"]["generator"])
    model["discriminator"].load_state_dict(state_dict["model"]["discriminator"])

    if optimizer is not None:
        optimizer["generator"].load_state_dict(state_dict["optimizer"]["generator"])
        optimizer["discriminator"].load_state_dict(state_dict["optimizer"]["discriminator"])

    if scheduler is not None:
        scheduler["generator"].load_state_dict(state_dict["scheduler"]["generator"])
        scheduler["discriminator"].load_state_dict(state_dict["scheduler"]["discriminator"])
    return state_dict['epoch'], state_dict['batch']


def save_checkpoint_wavegan(cfg, config_file, model, optimizer, scheduler, epoch_idx, batch_idx, output_folder_name='checkpoints'):
    """
    save model and parameters into a checkpoint file (.pth)
    :param cfg: the configuration object
    :param config_file: the configuration file path
    :param net: the network object
    :param optimizer: the optimizer object
    :param scheduler: the scheduler object
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
             'model': {
                "generator": model["generator"].state_dict(),
                "discriminator": model["discriminator"].state_dict(),
                },
             'optimizer': {
                "generator": optimizer["generator"].state_dict(),
                "discriminator": optimizer["discriminator"].state_dict(),
                },
             'scheduler': {
                "generator": scheduler["generator"].state_dict(),
                "discriminator": scheduler["discriminator"].state_dict(),
                },
            }
    torch.save(state, filename)
    # 用于在单卡和cpu上加载模型
    # torch.save(net.cpu().module.state_dict(), os.path.join(chk_folder, 'net_parameter.pkl'))
    shutil.copy(config_file, os.path.join(chk_folder, 'config.py'))
    