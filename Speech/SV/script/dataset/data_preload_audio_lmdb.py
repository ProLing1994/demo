import argparse
import lmdb
import librosa 
import numpy as np
import os
import pandas as pd
import sys
from pandas.io.parsers import read_csv
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/SV')
from utils.train_tools import load_cfg_file
from utils.folder_tools import *
from config.hparams import *
from dataset.audio import *


parser = argparse.ArgumentParser(description='Streamax SV Data Split Engine')
parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_TI_SV.py", help='config file')
args = parser.parse_args()

# params
args.commit_interval = 100


def general_lmdb(cfg, lmdb_path, csv_path, bool_background_audio=False):
    assert lmdb_path.endswith('.lmdb'), "[ERROR] lmdb_path must end with 'lmdb'."
    assert not os.path.exists(lmdb_path), "[ERROR] Folder [{:s}] already exists. Exit...".format(lmdb_path)

    # init
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)

    data_pd = pd.read_csv(csv_path)
    file_list = data_pd['file'].tolist()
    if not len(file_list):
        return

    # 估算映射空间大小（大概）
    data_size_per_audio = librosa.core.load(file_list[0], sr=sample_rate)[0].nbytes
    print('[Information] data size per audio is: ', data_size_per_audio)
    data_size = data_size_per_audio * len(file_list)
    
    # map_size：
    # Maximum size database may grow to; used to size the memory mapping. If database grows larger
    # than map_size, an exception will be raised and the user must close and reopen Environment.
    env = lmdb.open(lmdb_path, map_size=data_size * 10)
    txn = env.begin(write=True)

    drop_list = []
    for idx, row in tqdm(data_pd.iterrows(), total=len(file_list)):
        file_path = row['file']

        # key 
        key_byte = file_path.encode()

        # value
        if bool_background_audio:
            data = preprocess_wav(file_path, sample_rate, True, False)
        else:
            data = preprocess_wav(file_path, sample_rate, True, True)

        # check
        if len(data) == 0 or len(data) < desired_samples:
            drop_list.append(idx)
            continue

        txn.put(key_byte, data)

        if (idx + 1) % args.commit_interval == 0:
            txn.commit()
            # commit 之后需要再次 begin
            txn = env.begin(write=True)
    
    txn.commit()
    env.close()

    print("[Information] Drop wav: {}/{}".format(len(drop_list), len(data_pd)))
    data_pd.drop(drop_list, inplace=True)
    data_pd.to_csv(csv_path, index=False, encoding="utf_8_sig")


def preload_audio_lmdb():
    """ data preprocess engine
    :return:              None
    """
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # mkdir
    output_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_lmdb')
    create_folder(output_dir)

    # dataset
    for dataset_idx in range(len(cfg.general.TISV_dataset_list)):
        dataset_name = cfg.general.TISV_dataset_list[dataset_idx]

        print("Start preload dataset: {}: ".format(dataset_name))
        # init 
        lmdb_path = os.path.join(output_dir, '{}.lmdb'.format(dataset_name))
        csv_path = os.path.join(cfg.general.data_dir, dataset_name + '.csv')

        # general lmdb
        general_lmdb(cfg, lmdb_path, csv_path)
        print("Preload dataset:{}  Done!".format(dataset_name))


def preload_background_audio_lmdb():
    print("Start preload background audio: ")

    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # mkdir
    output_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_lmdb')
    create_folder(output_dir)

    # init 
    lmdb_path = os.path.join(output_dir, '{}.lmdb'.format(BACKGROUND_NOISE_DIR_NAME))
    csv_path = os.path.join(cfg.general.data_dir, 'background_noise_files.csv')

    # general lmdb
    general_lmdb(cfg, lmdb_path, csv_path, True)

    print("Preload background audio Done!")


def main():
    print("[Begin] Data Preload")
    preload_audio_lmdb()
    preload_background_audio_lmdb()
    print("[Done] Data Preload")


if __name__ == "__main__":
    main()