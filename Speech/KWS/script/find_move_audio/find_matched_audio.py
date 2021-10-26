import argparse
import os
import pandas as pd
import sys
import shutil
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_data_loader_pyimpl import WaveLoader_Librosa as WaveLoader

def find_matched_audio():

    for subfolder_idx in range(len(args.input_sudfolder_list)):
        input_dir = os.path.join(args.input_dir, args.input_sudfolder_list[subfolder_idx])
        output_dir = os.path.join(args.output_dir, args.output_sudfolder_list[subfolder_idx])

        # mkdir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(input_dir):
            continue

        # init 
        wave_list = os.listdir(input_dir)
        find_wave_list = os.listdir(args.find_input_dir)

        for idx in tqdm(range(len(wave_list))):
            wave_name = wave_list[idx].split('.')[0].split('_')[-1]
            wave_path = os.path.join(input_dir, wave_list[idx])

            find_bool = False
            for idy in range(len(find_wave_list)):
                find_wave_name = find_wave_list[idy].split('.')[0].split('_')[-1]
                # 匹配说话人和序列号（忽略设备id）
                if find_wave_name[:4] == wave_name[:4] and find_wave_name[-3:] == wave_name[-3:]:
                    find_bool = True
                    break

            if find_bool:
                output_path = os.path.join(output_dir, os.path.basename(wave_path).split('.')[0] + '.wav')
                print(wave_path, '->', output_path)
                shutil.move(wave_path, output_path)


if __name__ == "__main__":
    # 剔除错误数据
    default_input_dir = "/mnt/huanyuan/data/speech/Recording/MTA_Truck_Xiaoan/TruckIdling/mic_16k/"
    default_find_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_05132021/xiaoanxiaoan_8k_ignore/"
    default_output_dir = "/mnt/huanyuan/data/speech/Recording/MTA_Truck_Xiaoan/TruckIdling/mic_16k/"
    
    parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--find_input_dir', type=str, default=default_find_input_dir)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    args = parser.parse_args()

    # args.input_sudfolder_list = ['danbin_ori', 'danbin_asr', 'mic', 'adpro']
    # args.output_sudfolder_list = ['danbin_ori_once', 'danbin_asr_once', 'mic_once', 'adpro_once']
    # args.output_sudfolder_list = ['danbin_ori_ignore', 'danbin_asr_ignore', 'mic_ignore', 'adpro_ignore']
    args.input_sudfolder_list = ['xiaoanxiaoan_8k', 'xiaoanxiaoan_16k']
    args.output_sudfolder_list = ['xiaoanxiaoan_8k_ignore', 'xiaoanxiaoan_16k_ignore']
    find_matched_audio()

    # # 剔除超长语音数据
    # default_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_10182021/"
    # default_find_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_10182021/over_long/"
    # default_output_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_10182021/"
    
    # parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    # parser.add_argument('--input_dir', type=str, default=default_input_dir)
    # parser.add_argument('--find_input_dir', type=str, default=default_find_input_dir)
    # parser.add_argument('--output_dir', type=str, default=default_output_dir)
    # args = parser.parse_args()

    # args.input_sudfolder_list = ['xiaoanxiaoan_8k', 'xiaoanxiaoan_16k']
    # args.output_sudfolder_list = ['xiaoanxiaoan_8k_over_long', 'xiaoanxiaoan_16k_over_long']
    # find_matched_audio()

    # # 剔除声音较小语音数据
    # default_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_10182021/"
    # default_find_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_10182021/small_voice/"
    # default_output_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_10182021/"
    
    # parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    # parser.add_argument('--input_dir', type=str, default=default_input_dir)
    # parser.add_argument('--find_input_dir', type=str, default=default_find_input_dir)
    # parser.add_argument('--output_dir', type=str, default=default_output_dir)
    # args = parser.parse_args()

    # args.input_sudfolder_list = ['xiaoanxiaoan_8k', 'xiaoanxiaoan_16k']
    # args.output_sudfolder_list = ['xiaoanxiaoan_8k_small_voice', 'xiaoanxiaoan_16k_small_voice']
    # find_matched_audio()