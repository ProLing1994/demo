import argparse
import os
import pandas as pd
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.train_tools import load_cfg_file
from Basic.utils.folder_tools import *
from Basic.utils.hdf5_tools import *


def check(args, cfg, csv_path, dataset_name, mode_type):

    # init 
    drop_list = []
    data_pd = pd.read_csv(csv_path)
    hop_size = cfg.dataset.hop_size

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        dataset_name = data_pd.loc[idx, 'dataset']
        data_name = data_pd.loc[idx, 'unique_utterance']

        # wav
        wav_path = os.path.join(cfg.general.data_dir, 'dataset_audio_hdf5', dataset_name, data_name.split('.')[0] + '.h5')
        if not os.path.exists(wav_path):
            drop_list.append(idx)
            continue

        wav = read_hdf5(wav_path, "wave")

        # mel (T, C)
        mel = read_hdf5(wav_path, "feats")

        # Fix for missing padding   # TODO: settle on whether this is any useful
        r_pad =  (len(wav) // hop_size + 1) * hop_size - len(wav)
        wav = np.pad(wav, (0, r_pad), mode='constant')
        wav = wav[: len(mel) * hop_size]
        # make sure the audio length and feature length are matched
        if not len(mel) * hop_size == len(wav):
            drop_list.append(idx)
            continue

    drop_data_pd = data_pd.drop(drop_list) 
    drop_data_pd.to_csv(csv_path, index=False, encoding="utf_8_sig")

def data_check(args, mode_type):
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # dataset
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]

        print("Start preload dataset: {}, mode_type: {}".format(dataset_name, mode_type))

        # init 
        # csv_path = os.path.join(cfg.general.data_dir, dataset_name + '_' + mode_type + '.csv')
        csv_path = os.path.join(cfg.general.data_dir, dataset_name + '_' + mode_type + '_hdf5.csv')
        
        # generate hdf5
        check(args, cfg, csv_path, dataset_name, mode_type=mode_type) 
        print("Preload dataset:{}  Done!".format(dataset_name))

def main():
    parser = argparse.ArgumentParser(description='Streamax SV Data Split Engine')
    parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/TTS/config/tts/tts_config_chinese_sv2tts.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Data Check")
    data_check(args, hparams.TRAINING_NAME)
    print("[Done] Data Check")

if __name__ == "__main__":
    main()