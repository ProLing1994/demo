import argparse
import pandas as pd
import os 
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/SED')
from utils.folder_tools import *
from utils.train_tools import *

def create_indexes(args):
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # init 
    train_folder = os.path.join(cfg.general.data_dir, 'train')
    test_folder = os.path.join(cfg.general.data_dir, 'test')
    data_list = []         # {'label': [], 'file': [], 'mode': []}

    # create folder 
    output_dir = os.path.join(cfg.general.data_dir, '../experimental_dataset/dataset_{}_{}'.format(cfg.general.version, cfg.general.date))
    create_folder(output_dir)
    
    train_data_list = get_sub_filepaths_suffix(train_folder, '.wav')
    test_data_list = get_sub_filepaths_suffix(test_folder, '.wav')
    
    for idx in range(len(train_data_list)):
        wav_path = train_data_list[idx]
        wav_name = os.path.basename(wav_path)
        wav_label = wav_name.split('.')[0].split('-')[-1]
        data_list.append({'label': wav_label, 'file': wav_path, 'mode':'training'})

    for idx in range(len(test_data_list)):
        wav_path = test_data_list[idx]
        wav_name = os.path.basename(wav_path)
        wav_label = wav_name.split('.')[0].split('-')[-1]
        data_list.append({'label': wav_label, 'file': wav_path, 'mode':'testing'})

    data_pd = pd.DataFrame(data_list)
    data_pd.to_csv(cfg.general.data_csv_path, index=False, encoding="utf_8_sig")

if __name__ == '__main__':
    """
    功能描述：为数据库 ESC-50 提供训练集 和 测试集 train_test_dataset.csv
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--config_file', type=str, default='/home/huanyuan/code/demo/Speech/SED/config/sed_config_ESC50.py')

    # args = parser.parse_args()
    # create_indexes(args)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_create_indexes = subparsers.add_parser('create_indexes')
    parser_create_indexes.add_argument('-i', '--config_file', type=str, default='/home/huanyuan/code/demo/Speech/SED/config/sed_config_ESC50.py')
    parser_create_indexes.set_defaults(func=create_indexes)   
    args = parser.parse_args()
    
    args.func(args)
    