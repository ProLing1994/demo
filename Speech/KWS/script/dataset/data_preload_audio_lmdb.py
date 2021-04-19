import argparse
import lmdb
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from dataset.kws.dataset_helper import *
# from dataset.kws.kws_dataset import *
from utils.train_tools import load_cfg_file

parser = argparse.ArgumentParser(description='Streamax KWS Data Split Engine')
parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoan8k.py", help='config file')
parser.add_argument('-m', '--mode', type=str, default="training,validation,testing")
args = parser.parse_args()

# params
args.commit_interval = 100


def general_lmdb(cfg, lmdb_path, data_pd):
    assert lmdb_path.endswith('.lmdb'), "[ERROR] lmdb_path must end with 'lmdb'."
    assert not os.path.exists(lmdb_path), "[ERROR] Folder [{:s}] already exists. Exit...".format(lmdb_path)

    # init
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)

    file_list = data_pd['file'].tolist()
    label_list = data_pd['label'].tolist()

    if not len(file_list):
        return

    # 估算映射空间大小（大概）
    if label_list[0] == SILENCE_LABEL:
        data_size_per_audio = np.zeros(desired_samples, dtype=np.float32).nbytes
    else:
        data_size_per_audio = librosa.core.load(file_list[0], sr=sample_rate)[0].nbytes
    print('[Information] data size per audio is: ', data_size_per_audio)
    data_size = data_size_per_audio * len(file_list)
    
    # map_size：
    # Maximum size database may grow to; used to size the memory mapping. If database grows larger
    # than map_size, an exception will be raised and the user must close and reopen Environment.
    env = lmdb.open(lmdb_path, map_size=data_size * 10)
    txn = env.begin(write=True)

    for idx, row in tqdm(data_pd.iterrows(), total=len(label_list)):
        file_path = row['file']
        file_label = row['label']

        # key 
        key_byte = file_path.encode()

        # value
        if file_label == SILENCE_LABEL:
            data = np.zeros(desired_samples, dtype=np.float32)
        else:
            data = librosa.core.load(file_path, sr=sample_rate)[0]
        txn.put(key_byte, data)

        if (idx + 1) % args.commit_interval == 0:
            txn.commit()
            # commit 之后需要再次 begin
            txn = env.begin(write=True)
            
    txn.commit()
    env.close()


def preload_audio_lmdb(mode):
    """ data preprocess engine
    :param mode:  
    :return:              None
    """
    print("Start preload audio({}): ".format(mode))

    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # mkdir
    output_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date), 'dataset_audio_lmdb')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # init 
    lmdb_path = os.path.join(output_dir, '{}.lmdb'.format(mode))

    # load csv
    data_pd = pd.read_csv(cfg.general.data_csv_path)
    data_pd = data_pd[data_pd['mode'] == mode]

    # general lmdb
    general_lmdb(cfg, lmdb_path, data_pd)
    print("Preload audio({}) Done!".format(mode))


def preload_background_audio_lmdb():
    print("Start preload background audio: ")

    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # init
    sample_rate = cfg.dataset.sample_rate

    # mkdir
    output_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date), 'dataset_audio_lmdb')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # init 
    lmdb_path = os.path.join(output_dir, '{}.lmdb'.format(BACKGROUND_NOISE_DIR_NAME))

    # load csv
    background_data_pd = pd.read_csv(cfg.general.background_data_path)

    # general lmdb
    general_lmdb(cfg, lmdb_path, background_data_pd)

    print("Preload background audio Done!")


def main():
    print("[Begin] Data Preload")

    for mode in args.mode.split(','):
        preload_audio_lmdb(mode)

    preload_background_audio_lmdb()

    print("[Done] Data Preload")


if __name__ == "__main__":
    main()
