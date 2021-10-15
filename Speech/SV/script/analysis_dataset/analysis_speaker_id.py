import argparse
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.train_tools import *


def analysis_speaker_id(args):
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # load data_pd
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]
        csv_path = os.path.join(cfg.general.data_dir, dataset_name + '.csv')
    
        data_pd = pd.read_csv(csv_path)
        total_speaker_num = len(list(set(data_pd['speaker'].to_list())))

        data_pd_mode = data_pd[data_pd["mode"] == hparams.TRAINING_NAME]
        train_speaker_num = len(list(set(data_pd_mode['speaker'].to_list())))

        data_pd_mode = data_pd[data_pd["mode"] == hparams.TESTING_NAME]
        test_speaker_num = len(list(set(data_pd_mode['speaker'].to_list())))

        print("dataset: {}, speaker_num: {}/{}({})".format(dataset_name, train_speaker_num, test_speaker_num, total_speaker_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Streamax KWS Data Analysis Engine')
    # parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_english_TI_SV.py")
    parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_chinese_TI_SV.py")
    args = parser.parse_args()
    analysis_speaker_id(args)
    