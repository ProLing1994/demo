import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import sys
import torch

from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from common.common.utils.python.train_tools import EpochConcateSampler
from common.common.utils.python.plotly_tools import plot_loss4d, plot_loss2d, plot_loss

# sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
sys.path.insert(0, '/yuanhuan/code/demo/Speech')
from KWS.config.kws import hparams
# from KWS.dataset.kws.kws_dataset_align_preload_audio import SpeechDatasetAlign
from KWS.dataset.kws.kws_dataset_preload_audio_lmdb import SpeechDataset


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
        cfg.general.resume_epoch_num if cfg.general.resume_epoch_num != -1 else 0))
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
    if label_name_idx == hparams.SILENCE_LABEL:
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
