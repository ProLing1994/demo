import argparse
import pandas as pd
import os 
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/SED')
from utils.folder_tools import *
from utils.train_tools import *

def create_indexes(args):
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # init 
    data_list = []         # {'label': [], 'file': [], 'mode': []}
    dev_csv = os.path.join(cfg.general.data_dir, 'FSD50K.ground_truth', 'dev.csv')
    eval_csv = os.path.join(cfg.general.data_dir, 'FSD50K.ground_truth', 'eval.csv')
    vocabulary_csv = os.path.join(cfg.general.data_dir, 'FSD50K.ground_truth', 'vocabulary.csv')
    dev_folder = os.path.join(cfg.general.data_dir, 'FSD50K.dev_audio')
    eval_folder = os.path.join(cfg.general.data_dir, 'FSD50K.eval_audio')

    # create folder 
    output_dir = os.path.join(cfg.general.data_dir, '../experimental_dataset/dataset_{}_{}'.format(cfg.general.version, cfg.general.date))
    create_folder(output_dir)
    
    # label: vocabulary.csv
    label_dict = {}        # {'label_name': 'label_id'}
    vocabulary_pd = pd.read_csv(vocabulary_csv, names=['label_id', 'label_name', 'label_mid'])
    for _, row in vocabulary_pd.iterrows():
        label_dict[row['label_name']] = row['label_id']

    # dev.csv
    dev_pd = pd.read_csv(dev_csv)
    dev_data_list = get_sub_filepaths_suffix(dev_folder, '.wav')
    
    for idx in tqdm(range(len(dev_data_list))):
        wav_path = dev_data_list[idx]
        wav_name = os.path.basename(wav_path).split('.')[0]
        wav_pd = dev_pd[dev_pd['fname'] == int(wav_name)]
        wav_label_name = wav_pd['labels'].to_list()[0]
        wav_label_id = '/'.join([str(label_dict[label_name]) for label_name in wav_label_name.split(',')])
        wav_mode = 'training' if wav_pd['split'].to_list()[0] == 'train' else 'testing'
        data_list.append({'label': wav_label_id, 'file': wav_path, 'mode':wav_mode})

    # eval.csv
    eval_pd = pd.read_csv(eval_csv)
    eval_data_list = get_sub_filepaths_suffix(eval_folder, '.wav')

    for idx in tqdm(range(len(eval_data_list))):
        wav_path = eval_data_list[idx]
        wav_name = os.path.basename(wav_path).split('.')[0]
        wav_pd = eval_pd[eval_pd['fname'] == int(wav_name)]
        wav_label_name = wav_pd['labels'].to_list()[0]
        wav_label_id = '/'.join([str(label_dict[label_name]) for label_name in wav_label_name.split(',')])
        wav_mode = 'evaluate'
        data_list.append({'label': wav_label_id, 'file': wav_path, 'mode':wav_mode})

    data_pd = pd.DataFrame(data_list)
    data_pd.to_csv(cfg.general.data_csv_path, index=False, encoding="utf_8_sig")

if __name__ == '__main__':
    """
    功能描述：为数据库 FSD50K 提供训练集 和 测试集 train_test_dataset.csv
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--config_file', type=str, default='/home/huanyuan/code/demo/Speech/SED/config/sed_config_FSD50K.py')

    # args = parser.parse_args()
    # create_indexes(args)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_create_indexes = subparsers.add_parser('create_indexes')
    parser_create_indexes.add_argument('-i', '--config_file', type=str, default='/home/huanyuan/code/demo/Speech/SED/config/sed_config_FSD50K.py')
    parser_create_indexes.set_defaults(func=create_indexes)   
    args = parser.parse_args()
    
    args.func(args)
    