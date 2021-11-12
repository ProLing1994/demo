import argparse
import librosa
import os 
import pandas as pd
import sys

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.utils.train_tools import *

from KWS.config.kws import hparams
from KWS.dataset.kws.dataset_helper import *

def analysis_data_distribution(config_file):
    # load configuration file
    cfg = load_cfg_file(config_file)

    # init 
    sample_rate = cfg.dataset.sample_rate

    # load data 
    data_pd = pd.read_csv(cfg.general.data_csv_path)
    label_list = cfg.dataset.label.label_list

    # output_dir 
    output_dir = os.path.join(os.path.dirname(cfg.general.data_csv_path), 'data_distribution')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    data_number_dict = {}
    data_length_dict = {}
    avaliable_mode = ['training', 'validation', 'testing']
    for mode in avaliable_mode:
        data_number_dict[mode] = {}
        data_length_dict[mode] = {}
        mode_pd = data_pd[data_pd['mode'] == mode]
        for label in label_list:
            label_pd = mode_pd[mode_pd['label'] == label]
            data_number_dict[mode][label] = label_pd.shape[0]

            if label == hparams.SILENCE_LABEL:
                data_length_dict[mode][label] = 0.0
            else:
                audio_length_list = []
                file_list =  label_pd['file'].tolist()
                for file_path in tqdm(file_list):
                    audio_data = librosa.core.load(file_path, sr=sample_rate)[0]
                    audio_length = int(len(audio_data) * 1000 / sample_rate)
                    audio_length_list.append(audio_length)

                data_length_dict[mode][label] = "{:.2f}".format(np.array(audio_length_list).sum() / 1000.0 / 3600.0)
            
    print(data_number_dict)
    data_number_pd = pd.DataFrame(data_number_dict)
    data_number_pd.to_csv(os.path.join(output_dir, 'data_distribution.csv'))

    print(data_length_dict)
    data_length_pd = pd.DataFrame(data_length_dict)
    data_length_pd.to_csv(os.path.join(output_dir, 'data_length_distribution.csv'))


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py")
    # parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_2_label_xiaoyu.py")
    parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaole.py")
    args = parser.parse_args()

    print("[Begin] Analysis Audio Distribution")
    analysis_data_distribution(args.config_file)
    print("[Done] Analysis Audio Distribution")


if __name__ == "__main__":
    main()