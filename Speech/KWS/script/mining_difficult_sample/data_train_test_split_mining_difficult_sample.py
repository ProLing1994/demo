# -*- coding: utf-8 -*-
import argparse
import copy
import glob
import hashlib
import math
import os
import shutil
import pandas as pd
import random
import re
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *
from dataset.kws.dataset_helper import *
from script.dataset.data_train_test_split import *


def random_index(validation_percentage, testing_percentage):
    random_index = random.randint(0, 100)
    if random_index < validation_percentage:
        result = 'validation'
    elif random_index < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


def prepare_dataset_csv(config_file, input_dir, difficult_sample_mining_dir):
    """ data split engine
    :param config_file:   the input configuration file
    :return:              None
    """
    # load configuration file
    cfg = load_cfg_file(config_file)

    # set random seed 
    random.seed(RANDOM_SEED)

    # init 
    difficult_sample_mining = cfg.dataset.label.difficult_sample_mining
    difficult_sample_percentage = cfg.dataset.label.difficult_sample_percentage
    validation_percentage = cfg.dataset.label.validation_percentage
    testing_percentage = cfg.dataset.label.testing_percentage

    difficult_sample_files = []          # {'label': [], 'file': [], 'mode': []}
    simple_difficult_sample_files = []   # {'label': [], 'file': [], 'mode': []}
    # output
    output_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # input 
    if not os.path.exists(input_dir):
        input_dir = output_dir

    # 加载原有存在的 difficult_sample_files.csv
    if os.path.exists(os.path.join(output_dir, 'difficult_sample_files.csv')):
        difficult_sample_pd = pd.read_csv(os.path.join(output_dir, 'difficult_sample_files.csv'))
        for _, row in difficult_sample_pd.iterrows():
            difficult_sample_files.append({'label': row['label'], 'file': row['file'], 'mode':row['mode']})

    # Look through difficult sample to find audio samples
    search_path = os.path.join(difficult_sample_mining_dir, '*.wav')
    for wav_path in glob.glob(search_path):
        # Divide training, test and verification set
        set_index = random_index(validation_percentage, testing_percentage)
        difficult_sample_files.append({'label': UNKNOWN_WORD_LABEL, 'file': wav_path, 'mode':set_index})
        simple_difficult_sample_files.append({'label': UNKNOWN_WORD_LABEL, 'file': wav_path, 'mode':set_index})

    difficult_sample_pd = pd.DataFrame(difficult_sample_files)
    difficult_sample_pd.to_csv(os.path.join(output_dir, 'difficult_sample_files.csv'), index=False, encoding="utf_8_sig")

    # copy files
    if input_dir != output_dir:
        positive_data_csv = os.path.join(input_dir, 'positive_data_files.csv')
        unknown_csv = os.path.join(input_dir, 'unknown_files.csv')
        silence_csv = os.path.join(input_dir, 'silence_files.csv')
        background_noise_csv = os.path.join(input_dir, 'background_noise_files.csv')

        shutil.copy(positive_data_csv, os.path.join(output_dir, 'positive_data_files.csv'))
        shutil.copy(unknown_csv, os.path.join(output_dir, 'unknown_files.csv'))
        shutil.copy(silence_csv, os.path.join(output_dir, 'silence_files.csv'))
        shutil.copy(background_noise_csv, os.path.join(output_dir, 'background_noise_files.csv'))

    total_data_files = []             # {'label': [], 'file': [], 'mode': []}
    total_data_csv = os.path.join(input_dir, 'total_data_files.csv')
    if difficult_sample_mining == True:
        total_data_pd = pd.read_csv(total_data_csv)
        for _, row in total_data_pd.iterrows():
            total_data_files.append({'label': row['label'], 'file': row['file'], 'mode':row['mode']})
            
        # random
        random.shuffle(simple_difficult_sample_files)
        
        for set_index in ['validation', 'testing', 'training']:
            set_size = np.array([x['mode'] == set_index and x['label'] == cfg.dataset.label.positive_label[0] for x in total_data_files]).astype(np.int).sum()

            # difficult samples
            difficult_size = int(math.ceil(set_size * difficult_sample_percentage / 100))
            difficult_sample_files_set = [x for x in simple_difficult_sample_files if x['mode'] == set_index]
            difficult_sample_files_set = copy.deepcopy(difficult_sample_files_set)
            difficult_sample_files_set = difficult_sample_files_set[:difficult_size]
            for idx in range(len(difficult_sample_files_set)):
                difficult_sample_files_set[idx]['label'] = UNKNOWN_WORD_LABEL
            total_data_files.extend(difficult_sample_files_set)

        # random
        random.shuffle(total_data_files)

        total_data_pd = pd.DataFrame(total_data_files)
        total_data_pd.to_csv(os.path.join(output_dir, 'total_data_files.csv'), index=False, encoding="utf_8_sig")


def main():
    # We only add hard mining difficult negative samples, we do not change the distribution of the original data set, only add additional negative sample data
    parser = argparse.ArgumentParser(description='Streamax KWS Data Split Engine')
    # parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", help='config file')
    # parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_2_label_xiaoyu.py", help='config file')
    # parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaole.py", help='config file')
    parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui.py", help='config file')
    parser.add_argument('--input_dataset_dir', type=str, default="", help='The original dataset folder, if empty, will be changed under the current folder')
    parser.add_argument('--difficult_sample_mining_dir', type=str, default="/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_11032020/difficult_sample_mining_11122020/clean_audio/")
    args = parser.parse_args()

    print("[Begin] Add mining difficult sample")
    prepare_dataset_csv(args.config_file, args.input_dataset_dir, args.difficult_sample_mining_dir)
    print("[Done] Add mining difficult sample")

if __name__ == "__main__":
    main()
