import argparse
import os
from pickle import NONE
import pandas as pd
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/Basic')
from utils.folder_tools import *
from utils.train_tools import *

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/TTS')
from config.hparams import *


def load_dataset(dataset_name, dataset_path, data_files, mode, keep_speaker_ids=None):
    # load training dataset
    speaker_list = os.listdir(dataset_path)
    speaker_list.sort()
    
    if keep_speaker_ids: 
        speaker_list = [speaker_id for speaker_id in speaker_list if speaker_id in keep_speaker_ids]

    for speaker_idx in tqdm(range(len(speaker_list))):
        speaker_id = speaker_list[speaker_idx]

        section_list = os.listdir(os.path.join(dataset_path, speaker_id))
        section_list.sort()
        for section_idx in range(len(section_list)):
            section_idx = section_list[section_idx]

            utterance_list = os.listdir(os.path.join(dataset_path, speaker_id, section_idx))
            utterance_list.sort()
            for utterance_idx in range(len(utterance_list)):
                utterance_id = utterance_list[utterance_idx]
                if utterance_id.endswith('flac') or utterance_id.endswith('wav'):
                    file_path = os.path.join(os.path.join(dataset_path, speaker_id, section_idx, utterance_id))
                    data_files.append({'dataset': dataset_name, 'speaker': speaker_id, 'section': section_idx, 'utterance': section_idx + '_' + utterance_id, 'file': file_path, 'mode': mode})


def data_split_normal(cfg, dataset_name):
        dataset_training_path = cfg.general.dataset_path_dict[dataset_name+ "_training"] if dataset_name+ "_training" in cfg.general.dataset_path_dict else None
        dataset_testing_path = cfg.general.dataset_path_dict[dataset_name+ "_testing"] if dataset_name+ "_testing" in cfg.general.dataset_path_dict else None
        output_csv = os.path.join(cfg.general.data_dir, dataset_name + '.csv')

        if os.path.exists(output_csv):
            return 

        # init
        data_files = []                 # {'speaker': [], 'section': [], 'utterance': [], 'file': [], 'mode': []}
        
        print("[Begin] dataset: {}, set: {}".format(dataset_name, TRAINING_NAME))
        if dataset_training_path:
            load_dataset(dataset_name, dataset_training_path, data_files, TRAINING_NAME)
        print("[Begin] dataset: {}, set: {}".format(dataset_name, TESTING_NAME))
        if dataset_testing_path:
            load_dataset(dataset_name, dataset_testing_path, data_files, TESTING_NAME)

        data_pd = pd.DataFrame(data_files)
        data_pd.to_csv(os.path.join(cfg.general.data_dir, dataset_name + '.csv'), index=False, encoding="utf_8_sig")


def data_split(args):
    # load configuration file
    cfg = load_cfg_file(args.input)

    create_folder(cfg.general.data_dir)

    # dataset
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]

        data_split_normal(cfg, dataset_name)


def main():
    parser = argparse.ArgumentParser(description='Streamax TTS Data Split Engine')
    parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/tts_config_sv2tts.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Train test dataset split")
    data_split(args)
    print("[Done] Train test dataset split")


if __name__ == "__main__":
    main()
