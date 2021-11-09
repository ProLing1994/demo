import argparse
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.train_tools import load_cfg_file
from Basic.utils.folder_tools import *

from Basic.text.mandarin.pinyin.pinyin import get_pinyin
from Basic.text.mandarin.prosody.prosody import parse_cn_prosody_label_type1


def load_pinyin_aishell3(dataset_path, dataset_csv, mode):
    # text
    text_path = os.path.join(os.path.dirname(dataset_path), 'content.txt')
    assert os.path.exists(text_path), "[ERROR:] path: {}, do not exist.".format(text_path)

    with open(text_path, "r") as text_f:
        lines = [_line.split() for _line in text_f]

    text_dict = {}
    for _line in lines:
        text_dict[_line[0]]="".join(_line[1::2])

    # pd
    data_pd = pd.read_csv(dataset_csv)
    mode_data_pd = data_pd[data_pd['mode'] == mode]
    file_list = mode_data_pd['file'].tolist()

    for idx, row in tqdm(mode_data_pd.iterrows(), total=len(file_list)):
        utterance_id = row['utterance']
        
        # text
        text_key = str(utterance_id).split('_')[1]
        text_id = text_dict[text_key]
        
        mode_data_pd.loc[idx, 'text'] = text_id
        mode_data_pd.loc[idx, 'pinyin'] = " ".join(get_pinyin(text_id))
    
    # output csv
    mode_data_pd.to_csv(dataset_csv, index=False, encoding="utf_8_sig")

    return 
    

def load_pinyin_bansyp(dataset_path, dataset_csv, mode):
    # text
    text_path = os.path.join(os.path.dirname(dataset_path), 'ProsodyLabeling/000001-010000.txt')
    assert os.path.exists(text_path), "[ERROR:] path: {}, do not exist.".format(text_path)

    with open(text_path, "r") as text_f:
        lines = [_line.split() for _line in text_f]

    text_key = None
    text_dict = {}
    pinyin_dict = {}
    for line_idx in range(len(lines)):
        if line_idx % 2 == 0:
            text_key = lines[line_idx][0]
            text_dict[text_key] = lines[line_idx][1:][0]
        else:
            pinyin_dict[text_key] = " ".join(lines[line_idx])

    # pd 
    data_pd = pd.read_csv(dataset_csv)
    mode_data_pd = data_pd[data_pd['mode'] == mode]
    file_list = mode_data_pd['file'].tolist()

    for idx, row in tqdm(mode_data_pd.iterrows(), total=len(file_list)):
        utterance_id = row['utterance']
        
        # text
        text_key = str(utterance_id).split('_')[-1].split('.')[0]
        text_id = text_dict[text_key]
        pinyin_id = pinyin_dict[text_key]
        
        mode_data_pd.loc[idx, 'prosody'] = parse_cn_prosody_label_type1(text_id, pinyin_id)

        text_id = "".join(text_id.split('#1'))
        text_id = "".join(text_id.split('#2'))
        text_id = "".join(text_id.split('#3'))
        text_id = "".join(text_id.split('#4'))
        mode_data_pd.loc[idx, 'text'] = text_id
        mode_data_pd.loc[idx, 'pinyin'] = " ".join(get_pinyin(text_id))
    
    # output csv
    mode_data_pd.to_csv(dataset_csv, index=False, encoding="utf_8_sig")

    return 


def load_pinyin(dataset_name, dataset_path, dataset_csv, mode):
    # dataset_path == None, return
    if dataset_path == None:
        return 

    if not os.path.exists(dataset_csv):
        return 

    if dataset_name == 'Aishell3':
        load_pinyin_aishell3(dataset_path, dataset_csv, mode)
    elif dataset_name == 'BZNSYP':
        load_pinyin_bansyp(dataset_path, dataset_csv, mode)


def data_pinyin(args):
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # dataset
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]

        dataset_training_path = cfg.general.dataset_path_dict[dataset_name+ "_training"]
        dataset_testing_path = cfg.general.dataset_path_dict[dataset_name+ "_testing"]
        dataset_training_csv = os.path.join(cfg.general.data_dir, dataset_name + "_training" + '.csv')
        dataset_testing_csv = os.path.join(cfg.general.data_dir, dataset_name + "_testing" + '.csv')

        print("[Begin] dataset: {}, set: {}".format(dataset_name, hparams.TRAINING_NAME))
        load_pinyin(dataset_name, dataset_training_path, dataset_training_csv, hparams.TRAINING_NAME)
        print("[Begin] dataset: {}, set: {}".format(dataset_name, hparams.TESTING_NAME))
        load_pinyin(dataset_name, dataset_testing_path, dataset_testing_csv, hparams.TESTING_NAME)


def main():
    parser = argparse.ArgumentParser(description='Streamax SV Data Pinyin Engine')
    # parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/TTS/config/tts_config_english_sv2tts.py", help='config file')
    parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/TTS/config/sv2tts/tts_config_chinese_sv2tts.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Data Pinyin")
    data_pinyin(args)
    print("[Done] Data Pinyin")


if __name__ == "__main__":
    main()

