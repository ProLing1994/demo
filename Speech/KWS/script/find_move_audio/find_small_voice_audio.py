import argparse
import numpy as np 
import librosa
import os
import pandas as pd
import sys
import shutil
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_data_loader_pyimpl import WaveLoader_Librosa as WaveLoader
from Basic.utils.folder_tools import *

# 计算每一帧的能量 256个采样点为一帧
def cal_short_term_energy(wave_data) :
    energy = []
    sum = 0
    for i in range(len(wave_data)) :
        sum = sum + (float(wave_data[i]) * float(wave_data[i]))
        if (i + 1) % 256 == 0 :
            energy.append(sum)
            sum = 0
        # elif i == len(wave_data) - 1 :
        #     energy.append(sum)
    return energy

def find_small_voice_audio():
    # mkdir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # init 
    sample_rate = args.sample_rate
    wave_list = get_sub_filepaths_suffix(args.input_dir)

    for idx in tqdm(range(len(wave_list))):
        wave_path = wave_list[idx]
 
        # load data
        wave_data = librosa.core.load(wave_path, sr=sample_rate)[0]

        energy = cal_short_term_energy(wave_data)
        print(np.array(energy).max())
        if np.array(energy).max() < float(args.threshold):
            # output_path = os.path.join(args.output_dir, os.path.basename(wave_path).split('.')[0] + '.wav')
            output_subfolder_path = (os.path.dirname(wave_path) + '/').replace(args.input_dir, '')
            output_path = os.path.join(args.output_dir, output_subfolder_path, os.path.basename(wave_path).split('.')[0] + '.wav')
            create_folder(os.path.dirname(output_path))
            print(wave_path, '->', output_path)
            shutil.copy(wave_path, output_path)


if __name__ == "__main__":
    # # xiaoan 8k
    # default_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_10182021/xiaoanxiaoan_8k/"
    # default_output_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_10182021/small_voice/"
    # default_sample_rate = 8000
    # default_threshold = '0.2'

    # # activate bwc
    # default_input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/original_dataset/ActivateBWC_07162021/activatebwc/danbin_ori/"
    # default_output_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/original_dataset/ActivateBWC_07162021/activatebwc/small_voice/"
    # default_sample_rate = 16000
    # default_threshold = '0.02'

    # Gorila
    default_input_dir = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/original/office_TruckIdling/"
    default_output_dir = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/original/office_TruckIdling_small_voice/"
    default_sample_rate = 16000
    default_threshold = '0.05'

    parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    parser.add_argument('--threshold', type=str, default=default_threshold)
    parser.add_argument('--sample_rate', type=int, default=default_sample_rate)
    args = parser.parse_args()

    find_small_voice_audio()