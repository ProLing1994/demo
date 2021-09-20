import argparse
from pickle import NONE
import pandas as pd
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/SV')
from utils.train_tools import *
from utils.folder_tools import *
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
        dataset_training_path = cfg.general.TISV_dataset_path_dict[dataset_name+ "_training"]
        dataset_testing_path = cfg.general.TISV_dataset_path_dict[dataset_name+ "_testing"]
        output_csv = os.path.join(cfg.general.data_dir, dataset_name + '.csv')

        if os.path.exists(output_csv):
            return 

        # init
        data_files = []                 # {'speaker': [], 'section': [], 'utterance': [], 'file': [], 'mode': []}
        
        print("[Begin] dataset: {}, set: {}".format(dataset_name, TRAINING_NAME))
        load_dataset(dataset_name, dataset_training_path, data_files, TRAINING_NAME)
        print("[Begin] dataset: {}, set: {}".format(dataset_name, TESTING_NAME))
        load_dataset(dataset_name, dataset_testing_path, data_files, TESTING_NAME)

        data_pd = pd.DataFrame(data_files)
        data_pd.to_csv(os.path.join(cfg.general.data_dir, dataset_name + '.csv'), index=False, encoding="utf_8_sig")


def data_split_voxceleb1(cfg, dataset_name):
        dataset_training_path = cfg.general.TISV_dataset_path_dict[dataset_name+ "_training"]
        dataset_testing_path = cfg.general.TISV_dataset_path_dict[dataset_name+ "_testing"]
        dataset_csv_path = cfg.general.TISV_dataset_path_dict[dataset_name+ "_csv"]
        output_csv = os.path.join(cfg.general.data_dir, dataset_name + '.csv')
        
        if os.path.exists(output_csv):
            return 
            
        # Get the contents of the meta file
        with open(dataset_csv_path, "r") as metafile:
            metadata = [line.split("\t") for line in metafile][1:]
        
        # Select the ID and the nationality, filter out non-anglophone speakers
        nationalities = {line[0]: line[3] for line in metadata}
        keep_speaker_ids = [speaker_id for speaker_id, nationality in nationalities.items() if 
                            nationality.lower() in Anglophone_Nationalites]
        print("VoxCeleb1: using samples from %d (presumed anglophone) speakers out of %d." % 
            (len(keep_speaker_ids), len(nationalities)))

        # init
        data_files = []                 # {'dataset': [], 'speaker': [], 'section': [], 'utterance': [], 'file': [], 'mode': []}
        
        print("[Begin] dataset: {}, set: {}".format(dataset_name, TRAINING_NAME))
        load_dataset(dataset_name, dataset_training_path, data_files, TRAINING_NAME, keep_speaker_ids)
        print("[Begin] dataset: {}, set: {}".format(dataset_name, TESTING_NAME))
        load_dataset(dataset_name, dataset_testing_path, data_files, TESTING_NAME, keep_speaker_ids)

        data_pd = pd.DataFrame(data_files)
        data_pd.to_csv(os.path.join(cfg.general.data_dir, dataset_name + '.csv'), index=False, encoding="utf_8_sig")


def data_split_background_noise(cfg, dataset_name):
    # init
    background_noise_files = []           # {'label': [], 'file': []}
    background_noise_dir = cfg.general.TISV_dataset_path_dict[dataset_name]
    background_noise_list = os.listdir(background_noise_dir)
    background_noise_list.sort()

    output_csv = os.path.join(cfg.general.data_dir, 'background_noise_files.csv')

    if os.path.exists(output_csv):
        return 


    print("[Begin] dataset: {}".format(dataset_name))
    for background_noise_idx in tqdm(range(len(background_noise_list))):
        background_noise_id = background_noise_list[background_noise_idx]
        if background_noise_id.endswith('flac') or background_noise_id.endswith('wav'):
            background_noise_path = os.path.join(background_noise_dir, background_noise_id)
            background_noise_files.append({'label': BACKGROUND_NOISE_DIR_NAME, 'file':background_noise_path})

    background_noise_pd = pd.DataFrame(background_noise_files)
    background_noise_pd.to_csv(output_csv, index=False, encoding="utf_8_sig")


def data_split(args):
    # load configuration file
    cfg = load_cfg_file(args.input)

    create_folder(cfg.general.data_dir)

    # dataset
    for dataset_idx in range(len(cfg.general.TISV_dataset_list)):
        dataset_name = cfg.general.TISV_dataset_list[dataset_idx]

        if dataset_name == 'librispeech_other':
            data_split_normal(cfg, dataset_name)
        elif dataset_name == 'VoxCeleb1':
            data_split_voxceleb1(cfg, dataset_name)
        elif dataset_name == 'VoxCeleb2':
            data_split_normal(cfg, dataset_name)

    # background_noise dataset
    data_split_background_noise(cfg, "background_noise")


def main():
    parser = argparse.ArgumentParser(description='Streamax SV Data Split Engine')
    parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_TI_SV.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Train test dataset split")
    data_split(args)
    print("[Done] Train test dataset split")


if __name__ == "__main__":
    main()
