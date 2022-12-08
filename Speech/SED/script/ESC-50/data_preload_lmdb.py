import argparse
import librosa
import lmdb
import pandas as pd
import os 
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/SED')
from utils.folder_tools import *
from utils.train_tools import *

def general_lmdb(cfg, lmdb_path, data_pd):
    assert lmdb_path.endswith('.lmdb'), "[ERROR] LMDB path must end with 'lmdb'."
    assert not os.path.exists(lmdb_path), "[ERROR] Folder [{:s}] already exists. Exit...".format(lmdb_path)

    # init
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)

    file_list = data_pd['file'].tolist()
    label_list = data_pd['label'].tolist()
    
    assert len(file_list), "[ERROR] File .scv has no attribute：file. Exit..."

    # 估算映射空间大小（大概）
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
        data = librosa.core.load(file_path, sr=sample_rate)[0]
        txn.put(key_byte, data)

        if (idx + 1) % args.commit_interval == 0:
            txn.commit()
            # commit 之后需要再次 begin
            txn = env.begin(write=True)
            
    txn.commit()
    env.close()

def preload_audio_lmdb(args):

    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # mkdir
    output_dir = os.path.join(os.path.dirname(cfg.general.data_csv_path), 'dataset_audio_lmdb')
    create_folder(output_dir)

    # load csv
    data_pd = pd.read_csv(cfg.general.data_csv_path)

    # for
    for mode in args.mode.split(','):
        print("Start preload audio({}): ".format(mode))

        # general lmdb
        lmdb_path = os.path.join(output_dir, '{}.lmdb'.format(mode))
        data_pd_mode = data_pd[data_pd['mode'] == mode]
        general_lmdb(cfg, lmdb_path, data_pd_mode)

        print("Preload audio({}) Done!".format(mode))


if __name__ == '__main__':
    """
    功能描述：根据 train_test_dataset.csv，生成 lmdb 文件，用于模型训练和测试
    """

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--config_file', type=str, default='/home/huanyuan/code/demo/Speech/SED/config/sed_config_ESC50.py')
    # parser.add_argument('-m', '--mode', type=str, default="training,testing")
    # args = parser.parse_args()

    # # params
    # args.commit_interval = 100

    # preload_audio_lmdb(args)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_create_indexes = subparsers.add_parser('preload_audio_lmdb')
    parser_create_indexes.add_argument('-i', '--config_file', type=str, default='/home/huanyuan/code/demo/Speech/SED/config/sed_config_ESC50.py')
    parser_create_indexes.add_argument('-m', '--mode', type=str, default="training,testing")
    parser_create_indexes.set_defaults(func=preload_audio_lmdb)   
    args = parser.parse_args()
    
    # params
    args.commit_interval = 100

    args.func(args)