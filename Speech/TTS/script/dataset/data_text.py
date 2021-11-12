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


def load_text_BwcKeyword(dataset_path, dataset_csv, mode):
    # text
    text_path = os.path.join(os.path.dirname(dataset_path), '../transcript.txt')
    assert os.path.exists(text_path), "[ERROR:] path: {}, do not exist.".format(text_path)

    with open(text_path, "r") as text_f:
        lines = [_line.split() for _line in text_f]

    text_dict = {}
    for _line in lines:
        text_dict[_line[0]]=" ".join(_line[1:])

    # pd
    data_pd = pd.read_csv(dataset_csv)
    mode_data_pd = data_pd[data_pd['mode'] == mode]
    file_list = mode_data_pd['file'].tolist()

    for idx, row in tqdm(mode_data_pd.iterrows(), total=len(file_list)):
        utterance_id = row['utterance']
        
        # text
        text_key = "_".join(str(utterance_id).split('_')[2:])
        text_id = text_dict[text_key]
        
        mode_data_pd.loc[idx, 'text'] = text_id
    
    # output csv
    mode_data_pd.to_csv(dataset_csv, index=False, encoding="utf_8_sig")

    return 


def load_text(dataset_name, dataset_path, dataset_csv, mode):
    # dataset_path == None, return
    if dataset_path == None:
        return 

    if not os.path.exists(dataset_csv):
        return 

    if dataset_name == 'BwcKeyword':
        load_text_BwcKeyword(dataset_path, dataset_csv, mode)


def data_text(args):
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # dataset
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]

        dataset_training_path = cfg.general.dataset_path_dict[dataset_name+ "_training"] if dataset_name+ "_training" in cfg.general.dataset_path_dict else None 
        dataset_testing_path = cfg.general.dataset_path_dict[dataset_name+ "_testing"] if dataset_name+ "_testing" in cfg.general.dataset_path_dict else None
        dataset_training_csv = os.path.join(cfg.general.data_dir, dataset_name + "_training" + '.csv')
        dataset_testing_csv = os.path.join(cfg.general.data_dir, dataset_name + "_testing" + '.csv')

        print("[Begin] dataset: {}, set: {}".format(dataset_name, hparams.TRAINING_NAME))
        load_text(dataset_name, dataset_training_path, dataset_training_csv, hparams.TRAINING_NAME)
        print("[Begin] dataset: {}, set: {}".format(dataset_name, hparams.TESTING_NAME))
        load_text(dataset_name, dataset_testing_path, dataset_testing_csv, hparams.TESTING_NAME)


def main():
    parser = argparse.ArgumentParser(description='Streamax SV Data Text Engine')
    parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/TTS/config/sv2tts/tts_config_english_sv2tts.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Data Text")
    data_text(args)
    print("[Done] Data Text")


if __name__ == "__main__":
    main()

