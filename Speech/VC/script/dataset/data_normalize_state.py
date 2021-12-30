import argparse
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.utils.train_tools import load_cfg_file
from Basic.utils.folder_tools import *
from Basic.utils.hdf5_tools import *

from VC.dataset.cyclevae import audio_hdf5_normalize_state


def general_normalize_state(args, cfg, hdf5_dir, hdf5_normalize_dir, csv_path, dataset_name):

    data_pd = pd.read_csv(csv_path)
    file_list = data_pd['file'].tolist()

    if not len(file_list):
        print("[Information] file:{} empty. Exit...".format(csv_path))
        return

    audio_hdf5_normalize_state.preprocess_audio_hdf5_normalize_state(cfg, dataset_name, data_pd, hdf5_dir, hdf5_normalize_dir)


def data_normalize_state(args):
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # mkdir
    input_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_hdf5')
    output_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_normalize_hdf5')
    create_folder(output_dir)

    # dataset
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]

        print("Start preload dataset: {}".format(dataset_name))
        
        # init 
        hdf5_dir = os.path.join(input_dir, dataset_name)
        hdf5_normalize_dir = os.path.join(output_dir, dataset_name)
        csv_path = os.path.join(cfg.general.data_dir, dataset_name + '.csv')

        # mkdir
        create_folder(hdf5_normalize_dir)
        
        # general normalize state
        general_normalize_state(args, cfg, hdf5_dir, hdf5_normalize_dir, csv_path, dataset_name) 
        print("Preload dataset:{}  Done!".format(dataset_name))


def main():
    parser = argparse.ArgumentParser(description='Streamax SV Data Split Engine')
    parser.add_argument('-i', '--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/VC/config/cyclevae/vc_config_cyclevae.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Data Normalize State")
    data_normalize_state(args)
    print("[Done] Data Normalize State")


if __name__ == "__main__":
    main()