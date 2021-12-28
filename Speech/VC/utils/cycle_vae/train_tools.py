import sys
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/huanyuan/code/demo')
# sys.path.insert(0, '/yuanhuan/code/demo')
from common.common.utils.python.train_tools import EpochConcateSampler
from common.common.utils.python.plotly_tools import plot_loss4d, plot_loss2d, plot_loss

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech')
from Basic.utils.train_tools import *

from VC.dataset.cycle_vae.dataset_preload_audio_hdf5 import CycleVaeDataset


def generate_dataset_cycle_vae(cfg, mode):
    assert mode in ['training', 'testing', 'validation'], "[ERROR:] Unknow mode: {}".format(mode)
    
    dataset = CycleVaeDataset(cfg, mode)
    sampler = EpochConcateSampler(dataset, cfg.train.num_epochs - (cfg.general.resume_epoch_num if cfg.general.resume_epoch_num != -1 else 0))
    dataloader = DataLoader(dataset=dataset,
                            batch_size=cfg.train.batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.train.num_threads,
                            pin_memory=True)
    return dataloader, len(dataset)


def plot_tool_cycle_vae(cfg, log_file):
    """
    plot loss or accuracy
    :param cfg:                 config contain data set information
    :param log_file:            log_file
    """
    train_loss_file = os.path.join(cfg.general.save_dir, 'train_loss.html')
    epoch_train_loss_file = os.path.join(cfg.general.save_dir, 'epoch_train_loss.html')
    if cfg.general.is_test:
        plot_loss2d(log_file, train_loss_file, name=['train/batch_loss', 'eval/batch_loss'],
                    display='Training/Validation Loss', batch_word='iter')
        plot_loss2d(log_file, epoch_train_loss_file, name=['epoch_train/batch_loss', 'epoch_eval/batch_loss'],
                    display='Training/Validation Loss', batch_word='epoch')
    else:
        plot_loss(log_file, train_loss_file, name='train/batch_loss',
                display='Training Loss', batch_word='iter')
        plot_loss(log_file, epoch_train_loss_file, name='epoch_train/batch_loss',
                display='Training Loss', batch_word='epoch')


def save_checkpoint_cycle_vae(cfg, model, optimizer, epoch_idx, batch_idx, output_folder_name='checkpoints'):
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
                "encoder": model["encoder"].state_dict(),
                "decoder": model["decoder"].state_dict(),
                },
             'optimizer':  optimizer.state_dict(),
            }

    if is_pytorch_17plus:
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)

    # 用于在单卡和cpu上加载模型
    # torch.save(net.cpu().module.state_dict(), os.path.join(chk_folder, 'net_parameter.pkl'))