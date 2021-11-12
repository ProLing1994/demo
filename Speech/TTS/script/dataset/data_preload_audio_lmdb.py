import argparse
import lmdb
import librosa 
import numpy as np
import os
import pandas as pd
import sys
from pandas.io.parsers import read_csv
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.train_tools import load_cfg_file
from Basic.utils.folder_tools import *

from TTS.dataset.sv2tts import audio


parser = argparse.ArgumentParser(description='Streamax SV Data Split Engine')
parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/TTS/config/sv2tts/tts_config_english_sv2tts.py", help='config file')
# parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/TTS/config/sv2tts/tts_config_chinese_sv2tts.py", help='config file')
args = parser.parse_args()

# params
args.commit_interval = 100


def general_lmdb(cfg, lmdb_path, csv_path, dataset_name, mode_type='testing'):
    assert lmdb_path.endswith('.lmdb'), "[ERROR] lmdb_path must end with 'lmdb'."
    if os.path.exists(lmdb_path):
        print("[Information] Folder [{:s}] already exists. Exit...".format(lmdb_path))
        return 

    # init
    sample_rate = cfg.dataset.sample_rate

    data_pd = pd.read_csv(csv_path)
    mode_data_pd = data_pd[data_pd['mode'] == mode_type]

    file_list = mode_data_pd['file'].tolist()
    if not len(file_list):
        print("[Information] file:{} empty. Exit...".format(csv_path))
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

    # init
    data_files = []                 # {'dataset': [], 'speaker': [], 'section': [], 'utterance': [], 'file': [], text': [], 'unique_utterance', [], 'mode': []}

    for idx, row in tqdm(mode_data_pd.iterrows(), total=len(file_list)):

        metadata = audio.preprocess_speaker(cfg, data_files, row, dataset_name)
        
        for idy in range(len(metadata)):
            # key 
            key = metadata[idy][0]
            key_byte = key.encode()

            # value
            wav = metadata[idy][1]

            txn.put(key_byte, wav)

            if (idx + 1) % args.commit_interval == 0:
                txn.commit()
                # commit 之后需要再次 begin
                txn = env.begin(write=True)
            
            tqdm.write("[Information] Done key: {}, {}/{}".format(key, idx, len(file_list)))
    
    txn.commit()
    env.close()

    data_pd = pd.DataFrame(data_files) 
    out_put_csv = str(csv_path).split('.csv')[0] + '_' + mode_type + '.csv'
    data_pd.to_csv(out_put_csv, index=False, encoding="utf_8_sig")


def preload_audio_lmdb(mode_type):
    """ data preprocess engine
    :return:              None
    """
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # mkdir
    output_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_lmdb')
    create_folder(output_dir)

    # dataset
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]

        print("Start preload dataset: {}, mode_type: {}".format(dataset_name, mode_type))
        # init 
        lmdb_path = os.path.join(output_dir, '{}.lmdb'.format(dataset_name+'_'+mode_type))
        csv_path = os.path.join(cfg.general.data_dir, dataset_name + '.csv')

        # general lmdb
        general_lmdb(cfg, lmdb_path, csv_path, dataset_name, mode_type=mode_type) 
        print("Preload dataset:{}  Done!".format(dataset_name))


def main():
    print("[Begin] Data Preload")
    preload_audio_lmdb(hparams.TESTING_NAME)
    preload_audio_lmdb(hparams.TRAINING_NAME)
    print("[Done] Data Preload")


if __name__ == "__main__":
    main()