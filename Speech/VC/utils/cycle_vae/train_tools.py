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


def generate_test_dataset_cycle_vae(cfg, mode):
    """
    :param cfg:            config contain data set information
    :param mode:           Which partition to use, must be 'training', 'validation', or 'testing'.
    :param training_mode:  the model training mode, must be 0 or 1.
    :return:               data loader, length of data set
    """
    assert mode in ['training', 'testing', 'validation'], "[ERROR:] Unknow mode: {}".format(mode)

    dataset = CycleVaeDataset(cfg, mode)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              pin_memory=False,
                                              num_workers=1)
    return data_loader, len(dataset)


def plot_tool_cycle_vae(cfg, log_file):
    """
    plot loss or accuracy
    :param cfg:                 config contain data set information
    :param log_file:            log_file
    """
    train_loss_file = os.path.join(cfg.general.save_dir, 'train_loss.html')
    epoch_train_loss_file = os.path.join(cfg.general.save_dir, 'epoch_train_loss.html')
    epoch_rec_mcd_file = os.path.join(cfg.general.save_dir, 'epoch_rec_mcd.html')
    epoch_cv_mcd_file = os.path.join(cfg.general.save_dir, 'epoch_cv_mcd.html')
    epoch_lat_cosine_file = os.path.join(cfg.general.save_dir, 'epoch_lat_cosine.html')
    if cfg.general.is_test:
        plot_loss2d(log_file, train_loss_file, name=['train_iter_loss', 'eval_iter_loss'],
                    display='Training/Validation Loss', batch_word='iter')
        plot_loss2d(log_file, epoch_train_loss_file, name=['epoch_train_iter_loss', 'epoch_eval_iter_loss'],
                    display='Training/Validation Loss', batch_word='epoch')
        plot_loss2d(log_file, epoch_rec_mcd_file, name=['epoch_train_batch_mcd_src_src[0]', 'epoch_eval_batch_mcd_src_src[0]'],
                    display='Training/Validation Loss', batch_word='epoch')
        plot_loss2d(log_file, epoch_cv_mcd_file, name=['epoch_train_batch_mcd_src_trg[0]', 'epoch_eval_batch_mcd_src_trg[0]'],
                    display='Training/Validation Loss', batch_word='epoch')
        plot_loss2d(log_file, epoch_lat_cosine_file, name=['epoch_train_batch_lat_dist_cos_sim_src_trg[0]', 'epoch_eval_batch_lat_dist_cos_sim_src_trg[0]'],
                    display='Training/Validation Loss', batch_word='epoch')
    else:
        plot_loss(log_file, train_loss_file, name='train_iter_loss',
                display='Training Loss', batch_word='iter')
        plot_loss(log_file, epoch_train_loss_file, name='epoch_train_iter_loss',
                display='Training Loss', batch_word='epoch')
        plot_loss(log_file, epoch_rec_mcd_file, name='epoch_train_batch_mcd_src_src[0]',
                display='Training Loss', batch_word='epoch')
        plot_loss(log_file, epoch_cv_mcd_file, name='epoch_train_batch_mcd_src_trg[0]',
                display='Training Loss', batch_word='epoch')
        plot_loss(log_file, epoch_lat_cosine_file, name='epoch_train_batch_lat_dist_cos_sim_src_trg[0]',
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