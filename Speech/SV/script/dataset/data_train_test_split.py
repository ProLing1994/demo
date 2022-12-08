import argparse
from pickle import NONE
import pandas as pd
import re
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.train_tools import *
from Basic.utils.folder_tools import *

from SV.utils.train_tools import *


def load_dataset_type_1(dataset_name, dataset_path, data_files, mode, keep_speaker_ids=None):
    '''
    load_dataset
    数据格式：
    -- dataset_id
    -- | -- speaker_id
    -- | -- | -- section_id
    -- | -- | -- | -- utterance_id
    '''
    # load dataset
    speaker_list = os.listdir(dataset_path)
    speaker_list.sort()
    
    if keep_speaker_ids: 
        speaker_list = [speaker_id for speaker_id in speaker_list if speaker_id in keep_speaker_ids]

    for speaker_idx in tqdm(range(len(speaker_list))):
        speaker_id = speaker_list[speaker_idx]

        section_list = os.listdir(os.path.join(dataset_path, speaker_id))
        section_list.sort()
        for section_idx in range(len(section_list)):
            section_id = section_list[section_idx]

            utterance_list = os.listdir(os.path.join(dataset_path, speaker_id, section_id))
            utterance_list.sort()
            for utterance_idx in range(len(utterance_list)):
                utterance_id = utterance_list[utterance_idx]
                if utterance_id.endswith('flac') or utterance_id.endswith('wav'):
                    file_path = os.path.join(os.path.join(dataset_path, speaker_id, section_id, utterance_id))
                    data_files.append({'dataset': dataset_name, 'speaker': speaker_id, 'section': section_id, 'utterance': utterance_id, 'key': section_id + '_' + utterance_id, 'file': file_path, 'mode': mode})


def load_dataset_type_2(dataset_name, dataset_path, data_files, mode):
    '''
    load_dataset
    数据格式：
    -- speaker_id
    -- |-- utterance_id 
    '''
    # load dataset
    speaker_list = os.listdir(dataset_path)
    speaker_list.sort()

    # init 
    section_id = 0

    for speaker_idx in tqdm(range(len(speaker_list))):
        speaker_id = speaker_list[speaker_idx]

        speaker_path = os.path.join(dataset_path, speaker_id)
        if not os.path.isdir(speaker_path):
            continue

        utterance_list = os.listdir(speaker_path)
        utterance_list.sort()
        for utterance_idx in range(len(utterance_list)):
            utterance_id = utterance_list[utterance_idx]
            
            if utterance_id.endswith('flac') or utterance_id.endswith('wav'):
                file_path = os.path.join(os.path.join(dataset_path, speaker_id, utterance_id))
                data_files.append({'dataset': dataset_name, 'speaker': speaker_id, 'section': section_id, 'utterance': utterance_id, 'key': speaker_id + '_' + utterance_id, 'file': file_path, 'mode': mode})


def load_dataset_type_3(dataset_name, dataset_path, data_files, mode, dataset_format=None):
    '''
    load_dataset
    数据格式：
    -- wav 
    '''
    # load dataset
    wav_list = get_sub_filepaths_suffix(dataset_path, suffix='.wav')
    wav_list.sort()

    # init 
    section_id = 0
    
    for wav_idx in tqdm(range(len(wav_list))):
        file_path = wav_list[wav_idx]

        if dataset_format:
            # 20170001P00001A0001.wav
            # RM_Room_BWC_S1T1P1.wav
            file_name = os.path.basename(file_path).split('_')[-1]
            speaker_id = re.match(dataset_format, file_name).group(1)
            speaker_id = '{}_{}'.format(dataset_name, speaker_id)

        else:
            # 000001.wav
            speaker_id = '{}_{}'.format(dataset_name, '0')

        utterance_id = os.path.basename(file_path)

        data_files.append({'dataset': dataset_name, 'speaker': speaker_id, 'section': section_id, 'utterance': utterance_id, 'key': speaker_id + '_' + utterance_id, 'file': file_path, 'mode': mode})


def load_dataset_type_4(dataset_name, dataset_path, data_files, mode, dataset_format=None, dataset_subfolder=None):
    '''
    load_dataset
    数据格式：
    -- classification label 
    -- |-- wav 
    '''
    for subfolder_idx in range(len(dataset_subfolder)):
        subfolder = dataset_subfolder[subfolder_idx]

        # load dataset
        wav_list = get_sub_filepaths_suffix(os.path.join(dataset_path, subfolder), suffix='.wav')
        wav_list.sort()

        for wav_idx in tqdm(range(len(wav_list))):
            file_path = wav_list[wav_idx]

            if dataset_format:
                # 20170001P00001A0001.wav
                # RM_Room_BWC_S1T1P1.wav
                # file_name = os.path.basename(file_path).split('_')[-1]

                # 7275424M1_唤醒词_小鱼小鱼_女_中青年_否_0070.wav
                file_name = os.path.basename(file_path)
                
                match = re.match(dataset_format, file_name)
                if not match:
                    continue

                speaker_id = match.group(1)
                speaker_id = '{}_{}'.format(dataset_name, speaker_id)

            else:
                # 000001.wav
                speaker_id = '{}_{}'.format(dataset_name, '0')

            section_id = 0
            utterance_id = os.path.basename(file_path)

            data_files.append({'dataset': dataset_name, 'speaker': speaker_id, 'section': section_id, 'utterance': utterance_id, 'key': speaker_id + '_' + utterance_id, 'file': file_path, 'mode': mode})


def load_dataset_normal(dataset_name, dataset_path, data_files, mode, keep_speaker_ids=None, type=1, dataset_format=None, dataset_subfolder=None):
    # dataset_path == None, return
    if dataset_path == None:
        return 

    if type == 1:
        load_dataset_type_1(dataset_name, dataset_path, data_files, mode, keep_speaker_ids=keep_speaker_ids)
    if type == 2:
        load_dataset_type_2(dataset_name, dataset_path, data_files, mode)
    if type == 3:
        load_dataset_type_3(dataset_name, dataset_path, data_files, mode, dataset_format=dataset_format)
    if type == 4:
        load_dataset_type_4(dataset_name, dataset_path, data_files, mode, dataset_format=dataset_format, dataset_subfolder=dataset_subfolder)


def load_dataset_csv(dataset_name, dataset_csv_path, dataset_format, data_files, mode):
    '''
    load_dataset
    数据格式：
    -- 根据 csv 文件自动分割
    '''
    # dataset_path == None, return
    if dataset_csv_path == None:
        return 

    dataset_pd = pd.read_csv(dataset_csv_path)
    dataset_pd = dataset_pd[dataset_pd['mode'] == mode]

    for _, row in tqdm(dataset_pd.iterrows()):
        file_path = row['file']
        file_name = os.path.basename(file_path).split('_')[-1]

        speaker_id = re.match(dataset_format, file_name).group(1)
        speaker_id = '{}_{}'.format(dataset_name, speaker_id)

        section_id = 0
        utterance_id = file_name
        data_files.append({'dataset': dataset_name, 'speaker': speaker_id, 'section': section_id, 'utterance': utterance_id, 'file': file_path, 'mode': mode})


def data_split_normal(cfg, dataset_name, type=1):
    '''
    data_split_normal
    '''
    dataset_training_path = cfg.general.dataset_path_dict[dataset_name+ "_training"] if dataset_name+ "_training" in cfg.general.dataset_path_dict else None
    dataset_testing_path = cfg.general.dataset_path_dict[dataset_name+ "_testing"] if dataset_name+ "_testing" in cfg.general.dataset_path_dict else None
    dataset_format = cfg.general.dataset_path_dict[dataset_name+ "_format"] if dataset_name+ "_format" in cfg.general.dataset_path_dict else None
    dataset_subfolder = cfg.general.dataset_path_dict[dataset_name+ "_subfolder"] if dataset_name+ "_subfolder" in cfg.general.dataset_path_dict else None
    output_csv = os.path.join(cfg.general.data_dir, dataset_name + '.csv')

    if os.path.exists(output_csv):
        return 

    # init
    data_files = []                 # {'speaker': [], 'section': [], 'utterance': [], 'key': [], 'file': [], 'mode': []}
    
    print("[Begin] dataset: {}, set: {}".format(dataset_name, hparams.TRAINING_NAME))
    load_dataset_normal(dataset_name, dataset_training_path, data_files, hparams.TRAINING_NAME, type=type, dataset_format=dataset_format, dataset_subfolder=dataset_subfolder)
    print("[Begin] dataset: {}, set: {}".format(dataset_name, hparams.TESTING_NAME))
    load_dataset_normal(dataset_name, dataset_testing_path, data_files, hparams.TESTING_NAME, type=type, dataset_format=dataset_format, dataset_subfolder=dataset_subfolder)

    data_pd = pd.DataFrame(data_files)
    data_pd.to_csv(output_csv, index=False, encoding="utf_8_sig")


def data_split_csv(cfg, dataset_name):
    '''
    data_split_csv
    '''
    dataset_csv_path = os.path.join(cfg.general.dataset_path_dict[dataset_name], 'positive_data_files.csv')
    dataset_format = cfg.general.dataset_path_dict[dataset_name+ "_format"]
    output_csv = os.path.join(cfg.general.data_dir, dataset_name + '.csv')

    if os.path.exists(output_csv):
        return 
    
    # init
    data_files = []                 # {'speaker': [], 'section': [], 'utterance': [], 'file': [], 'mode': []}

    print("[Begin] dataset: {}, set: {}".format(dataset_name, hparams.TRAINING_NAME))
    load_dataset_csv(dataset_name, dataset_csv_path, dataset_format, data_files, hparams.TRAINING_NAME)
    print("[Begin] dataset: {}, set: {}".format(dataset_name, hparams.TESTING_NAME))
    load_dataset_csv(dataset_name, dataset_csv_path, dataset_format, data_files, hparams.TESTING_NAME)
    print("[Begin] dataset: {}, set: {}".format(dataset_name, hparams.VALIDATION_NAME))
    load_dataset_csv(dataset_name, dataset_csv_path, dataset_format, data_files, hparams.VALIDATION_NAME)

    data_pd = pd.DataFrame(data_files)
    data_pd.to_csv(output_csv, index=False, encoding="utf_8_sig")


def data_split_voxceleb1(cfg, dataset_name, type=1):
    '''
    data_split_voxceleb1：voxceleb1 特殊版本
    说明：
        voxceleb1 中包含 vox1_meta.csv，用于挑选不同国家的数据集
    '''
    dataset_training_path = cfg.general.dataset_path_dict[dataset_name+ "_training"]
    dataset_testing_path = cfg.general.dataset_path_dict[dataset_name+ "_testing"]
    dataset_csv_path = cfg.general.dataset_path_dict[dataset_name+ "_csv"]
    output_csv = os.path.join(cfg.general.data_dir, dataset_name + '.csv')
    
    if os.path.exists(output_csv):
        return 
        
    # Get the contents of the meta file
    with open(dataset_csv_path, "r") as metafile:
        metadata = [line.split("\t") for line in metafile][1:]
    
    # Select the ID and the nationality, filter out non-anglophone speakers
    nationalities = {line[0]: line[3] for line in metadata}
    keep_speaker_ids = [speaker_id for speaker_id, nationality in nationalities.items() if 
                        nationality.lower() in hparams.Anglophone_Nationalites]
    print("VoxCeleb1: using samples from %d (presumed anglophone) speakers out of %d." % 
        (len(keep_speaker_ids), len(nationalities)))

    # init
    data_files = []                 # {'dataset': [], 'speaker': [], 'section': [], 'utterance': [], 'file': [], 'mode': []}
    
    print("[Begin] dataset: {}, set: {}".format(dataset_name, hparams.TRAINING_NAME))
    load_dataset_normal(dataset_name, dataset_training_path, data_files, hparams.TRAINING_NAME, keep_speaker_ids=keep_speaker_ids, type=type)
    print("[Begin] dataset: {}, set: {}".format(dataset_name, hparams.TESTING_NAME))
    load_dataset_normal(dataset_name, dataset_testing_path, data_files, hparams.TESTING_NAME, keep_speaker_ids=keep_speaker_ids, type=type)

    data_pd = pd.DataFrame(data_files)
    data_pd.to_csv(output_csv, index=False, encoding="utf_8_sig")


def data_split_background_noise(cfg, dataset_name):
    if not dataset_name in cfg.general.dataset_path_dict:
        return 

    # init
    background_noise_files = []           # {'label': [], 'file': []}
    background_noise_dir = cfg.general.dataset_path_dict[dataset_name]
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
            background_noise_files.append({'label': hparams.BACKGROUND_NOISE_DIR_NAME, 'file':background_noise_path})

    background_noise_pd = pd.DataFrame(background_noise_files)
    background_noise_pd.to_csv(output_csv, index=False, encoding="utf_8_sig")


def data_split(args):
    # load configuration file
    cfg = load_cfg_file(args.input)

    create_folder(cfg.general.data_dir)

    # dataset
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]

        if dataset_name in ['librispeech_clean_100', 'librispeech_clean_360', 'librispeech_test_clean', 'librispeech_other']:
            data_split_normal(cfg, dataset_name, type = 1)
        elif dataset_name in ['VoxCeleb1']:
            data_split_voxceleb1(cfg, dataset_name, type = 1)
        elif dataset_name in ['Aishell3', 'SLR62', 'SLR68', 'CN-Celeb1', 'CN-Celeb2', 'VoxCeleb2', 'VCC2020', 'BZNSYP_Aishell3']:
            data_split_normal(cfg, dataset_name, type = 2)
        elif dataset_name in ['SLR38', 'BZNSYP', 'BwcKeyword']:
            data_split_normal(cfg, dataset_name, type = 3)
        elif dataset_name in ['XiaoYu']:
            data_split_normal(cfg, dataset_name, type = 4)
        elif dataset_name in ["XiaoRui", "XiaoAn"]:
            data_split_csv(cfg, dataset_name)

    # background_noise dataset
    data_split_background_noise(cfg, "background_noise")


def main():
    parser = argparse.ArgumentParser(description='Streamax SV Data Split Engine')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_english_TI_SV.py", help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_chinese_TI_SV.py", help='config file')
    parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_chinese_TD_SV.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Train test dataset split")
    data_split(args)
    print("[Done] Train test dataset split")


if __name__ == "__main__":
    main()
