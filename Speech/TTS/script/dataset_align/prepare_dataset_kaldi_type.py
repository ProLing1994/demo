import argparse
import os
import re
import pandas as pd
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.train_tools import load_cfg_file
from Basic.utils.folder_tools import *


def dataset_kaldi_type(cfg, dataset_name, dataset_csv, mode):

    if not os.path.exists(dataset_csv):
        return 

    dest_dir = os.path.join(cfg.general.data_dir, 'dataset_align', dataset_name + "_" + mode)
    create_folder(dest_dir)

    file_text = open(dest_dir + "/text", "w")
    file_wav = open(dest_dir + "/wav.scp", "w")
    file_spk = open(dest_dir + "/utt2spk", "w")

    # pd
    data_pd = pd.read_csv(dataset_csv)
    mode_data_pd = data_pd[data_pd['mode'] == mode]
    file_list = mode_data_pd['file'].tolist()

    for _, row in tqdm(mode_data_pd.iterrows(), total=len(file_list)):
        utterance = row["utterance"]
        file = row["file"]
        text = row["text"]
        speaker = row["speaker"]

        # text，去掉无用字符
        text = "".join(text.split('#1'))
        text = "".join(text.split('#2'))
        text = "".join(text.split('#3'))
        text = "".join(text.split('#4'))

        # text，去掉无用字符
        text = re.sub('[“”、，。：；？！—…#（）%$]', '', text)
        text = text.strip()

        # text 添加空格，否则 kaldi 换转换为 OOV
        res_text = ""
        for idx in range(len(text)):
            res_text += text[idx] + " "

        file_text.writelines(utterance + " " + res_text + "\n")
        file_wav.writelines(utterance + " " + file + "\n")
        file_spk.writelines(utterance + " " + speaker + "\n" )

    file_text.close()
    file_wav.close()
    file_spk.close()


def prepare_dataset_kaldi_type(args):

    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # dataset
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]
        dataset_training_csv = os.path.join(cfg.general.data_dir, dataset_name + "_training" + '.csv')
        dataset_testing_csv = os.path.join(cfg.general.data_dir, dataset_name + "_testing" + '.csv')

        print("[Begin] dataset: {}, set: {}".format(dataset_name, hparams.TRAINING_NAME))
        dataset_kaldi_type(cfg, dataset_name, dataset_training_csv, hparams.TRAINING_NAME)
        print("[Begin] dataset: {}, set: {}".format(dataset_name, hparams.TESTING_NAME))
        dataset_kaldi_type(cfg, dataset_name, dataset_testing_csv, hparams.TESTING_NAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/TTS/config/sv2tts/tts_config_chinese_sv2tts.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Data Pinyin")
    prepare_dataset_kaldi_type(args)
    print("[Done] Data Pinyin")


if __name__ == "__main__":
    main()