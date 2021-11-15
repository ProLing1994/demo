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

    # init
    train_speaker_list = []
    test_speaker_list = []
    total_speaker_list = []

    train_utterance_list = []
    test_utterance_list = []
    total_utterance_list = []

    # load data_pd
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]
        csv_path = os.path.join(cfg.general.data_dir, dataset_name + '.csv')

        data_pd = pd.read_csv(csv_path)

        data_pd_mode = data_pd[data_pd["mode"] == hparams.TRAINING_NAME]
        train_speaker_num = len(list(set(data_pd_mode['speaker'].to_list())))
        train_utterancer_num = len(list(set(data_pd_mode['utterance'].to_list())))
        train_speaker_list.extend(list(set(data_pd_mode['speaker'].to_list())))
        train_utterance_list.extend(list(set(data_pd_mode['utterance'].to_list())))

        data_pd_mode = data_pd[data_pd["mode"] == hparams.TESTING_NAME]
        test_speaker_num = len(list(set(data_pd_mode['speaker'].to_list())))
        test_utterance_num = len(list(set(data_pd_mode['utterance'].to_list())))
        test_speaker_list.extend(list(set(data_pd_mode['speaker'].to_list())))
        test_utterance_list.extend(list(set(data_pd_mode['utterance'].to_list())))

        total_speaker_num = len(list(set(data_pd['speaker'].to_list())))
        total_utterance_num = len(list(set(data_pd['utterance'].to_list())))
        total_speaker_list.extend(list(set(data_pd['speaker'].to_list())))
        total_utterance_list.extend(list(set(data_pd['utterance'].to_list())))

        print("dataset: {}, speaker num: {}/{}({}), utterancer num: {}/{}({})".format(dataset_name, train_speaker_num, test_speaker_num, total_speaker_num,
                                                                                        train_utterancer_num, test_utterance_num, total_utterance_num))
    print("total speaker num: {}/{}({}), utterancer num: {}/{}({})".format(len(train_speaker_list), len(test_speaker_list), len(total_speaker_list), 
                                                                                len(train_utterance_list), len(test_utterance_list), len(total_utterance_list)))
    print("total set speaker num: {}/{}({}), utterancer num: {}/{}({})".format(len(list(set(train_speaker_list))), len(list(set(test_speaker_list))), len(list(set(total_speaker_list))),
                                                                                len(list(set(train_utterance_list))), len(list(set(test_utterance_list))), len(list(set(total_utterance_list))) ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Streamax KWS Data Analysis Engine')
    # parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_english_TI_SV.py")
    # parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_chinese_TI_SV.py")
    parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_chinese_TD_SV.py")
    args = parser.parse_args()
    analysis_speaker_id(args)
    