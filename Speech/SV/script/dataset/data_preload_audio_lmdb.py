import argparse
import lmdb
import librosa 
import os
import pandas as pd
import sys
from pandas.io.parsers import read_csv
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.train_tools import load_cfg_file
from Basic.utils.folder_tools import *


parser = argparse.ArgumentParser(description='Streamax SV Data preload Engine')
# parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_english_TI_SV.py", help='config file')
# parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_chinese_TI_SV.py", help='config file')
parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_chinese_TD_SV.py", help='config file')
args = parser.parse_args()

# params
args.commit_interval = 100


def general_lmdb(cfg, lmdb_path, csv_path, mode_type='testing', bool_background_audio=False):
    assert lmdb_path.endswith('.lmdb'), "[ERROR] lmdb_path must end with 'lmdb'."
    if os.path.exists(lmdb_path):
        print("[Information] Folder [{:s}] already exists. Exit...".format(lmdb_path))
        return 

    # init
    sample_rate = cfg.dataset.sample_rate
    desired_samples = int(cfg.dataset.sample_rate * hparams.check_wave_length_ms / 1000)

    data_pd = pd.read_csv(csv_path)
    if bool_background_audio:
        mode_data_pd = data_pd
    else:
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
    drop_list = []

    for idx, row in tqdm(mode_data_pd.iterrows(), total=len(file_list)):
        file_path = row['file']

        # key 
        key_byte = file_path.encode()

        # value
        try:
            data = librosa.core.load(file_path, sr=sample_rate)[0]
        except:
            drop_list.append(idx)
            continue

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
        general_lmdb(cfg, lmdb_path, csv_path, mode_type=mode_type)
        print("Preload dataset:{}  Done!".format(dataset_name))


def preload_background_audio_lmdb():
    print("Start preload background audio: ")

    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # mkdir
    output_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_lmdb')
    create_folder(output_dir)

    # init 
    lmdb_path = os.path.join(output_dir, '{}.lmdb'.format(hparams.BACKGROUND_NOISE_DIR_NAME))
    csv_path = os.path.join(cfg.general.data_dir, 'background_noise_files.csv')

    # general lmdb
    general_lmdb(cfg, lmdb_path, csv_path, bool_background_audio=True)

    print("Preload background audio Done!")


def main():
    print("[Begin] Data Preload")
    preload_audio_lmdb(hparams.TRAINING_NAME)
    preload_audio_lmdb(hparams.TESTING_NAME)
    preload_audio_lmdb(hparams.VALIDATION_NAME)
    preload_background_audio_lmdb()
    print("[Done] Data Preload")


if __name__ == "__main__":
    main()