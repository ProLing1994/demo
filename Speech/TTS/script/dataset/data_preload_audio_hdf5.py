import argparse
import pandas as pd
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.train_tools import load_cfg_file
from Basic.utils.folder_tools import *

from TTS.dataset.tts import audio_hdf5


def generate_hdf5(args, cfg, hdf5_dir, csv_path, dataset_name, mode_type='testing'):

    data_pd = pd.read_csv(csv_path)
    mode_data_pd = data_pd[data_pd['mode'] == mode_type]

    file_list = mode_data_pd['file'].tolist()
    if not len(file_list):
        print("[Information] file:{} empty. Exit...".format(csv_path))
        return

    # init
    data_lists = []                 # {'dataset': [], 'speaker': [], 'section': [], 'utterance': [], 'file': [], text': [], 'unique_utterance', [], 'mode': []}

    for idx, row in tqdm(mode_data_pd.iterrows(), total=len(file_list)):

        audio_hdf5.preprocess_audio_hdf5(cfg, row, dataset_name, hdf5_dir, data_lists)

    data_pd = pd.DataFrame(data_lists) 
    out_put_csv = str(csv_path).split('.csv')[0] + '_' + mode_type + '_hdf5.csv'
    data_pd.to_csv(out_put_csv, index=False, encoding="utf_8_sig")


def preload_audio_hdf5(args, mode_type):
    """ data preprocess engine
    :return:              None
    """
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # mkdir
    output_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_hdf5')
    create_folder(output_dir)

    # dataset
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]

        print("Start preload dataset: {}, mode_type: {}".format(dataset_name, mode_type))
        # init 
        hdf5_dir = os.path.join(output_dir, dataset_name)
        csv_path = os.path.join(cfg.general.data_dir, dataset_name + '.csv')

        # mkdir
        create_folder(hdf5_dir)
        
        # generate hdf5
        generate_hdf5(args, cfg, hdf5_dir, csv_path, dataset_name, mode_type=mode_type) 
        print("Preload dataset:{}  Done!".format(dataset_name))


def main():
    parser = argparse.ArgumentParser(description='Streamax SV Data Split Engine')
    parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/TTS/config/tts/tts_config_chinese_sv2tts.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Data Preload")
    preload_audio_hdf5(args, hparams.TESTING_NAME)
    preload_audio_hdf5(args, hparams.TRAINING_NAME)
    print("[Done] Data Preload")


if __name__ == "__main__":
    main()