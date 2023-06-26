import argparse
import os
import sys

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.utils.folder_tools import *

def remove_audio_samename():
    wave_list = get_sub_filepaths_suffix(args.input_dir, '.wav')

    for idx in tqdm(range(len(wave_list))):
        wave_path = wave_list[idx]
        output_subfolder_path = (os.path.dirname(wave_path) + '/').replace(args.input_dir, '')
        output_path = os.path.join(args.output_dir, output_subfolder_path, os.path.basename(wave_path).split('.')[0] + '.wav')
        
        if os.path.exists(output_path):
            print(output_path)
            # os.remove(output_path)


def remove_audio_suffix():
    path_list = get_sub_filepaths_suffix(args.input_dir, args.suffix)

    for idx in tqdm(range(len(path_list))):
        if os.path.exists(path_list[idx]):
            print(path_list[idx])
            # os.remove(path_list[idx])


def remove_audio_speaker():
    wave_list = get_sub_filepaths_suffix(args.input_dir, '.wav')

    for idx in tqdm(range(len(wave_list))):
        wave_path = wave_list[idx]
        wave_name = os.path.basename(wave_path)

        for idy in range(len(args.spk_list)):
            spk_idx = args.spk_list[idy]

            if spk_idx in wave_name:

                print(wave_path)
                # os.remove(wave_path)


if __name__ == "__main__":

    # # 方法：删除同名文件
    # parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    # args = parser.parse_args()
    # # args.input_dir = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/TruckIdling_test_over_long/"
    # # args.output_dir = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/TruckIdling_test/"
    # args.input_dir = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/office_test_over_long/"
    # args.output_dir = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/office_test/"

    # remove_audio_samename()

    # # 方法：删除指定后缀
    # parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    # args = parser.parse_args()
    # args.input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/original_dataset/ActivateBWC_07162021/activatebwc/原始音频/"
    # args.suffix = '_temp.wav'

    # remove_audio_suffix()

    # 方法：删除指定说话人
    parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    args = parser.parse_args()
    # args.input_dir = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/TruckIdling_test/"
    args.input_dir = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/office_test/"
    args.spk_list = ['S043', 'S076', 'S047', 'S050', 'S084', 'S069', 'S080', 'S046', 'S013']

    remove_audio_speaker()