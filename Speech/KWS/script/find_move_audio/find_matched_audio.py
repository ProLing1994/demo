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
    # # 剔除错误数据
    # default_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_05132021/xiaoan_0422/error/"
    # # default_find_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_05132021/xiaoan_0422/error/once/"
    # default_find_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_05132021/xiaoan_0422/error/ignore/"
    # default_output_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_05132021/xiaoan_0422/error/"
    
    # parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    # parser.add_argument('--input_dir', type=str, default=default_input_dir)
    # parser.add_argument('--find_input_dir', type=str, default=default_find_input_dir)
    # parser.add_argument('--output_dir', type=str, default=default_output_dir)
    # args = parser.parse_args()

    # # args.input_sudfolder_list = ['danbin_ori', 'danbin_asr', 'mic', 'adpro']
    # # args.output_sudfolder_list = ['16k_once', '16k_once', '16k_once', '8k_once']
    # args.input_sudfolder_list = ['danbin_ori', 'danbin_asr', 'mic', 'adpro']
    # args.output_sudfolder_list = ['16k_ignore', '16k_ignore', '16k_ignore', '8k_ignore']
    # find_matched_audio()

    # 剔除超长语音数据
    default_input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/original_dataset/ActivateBWC_07162021/activatebwc/"
    default_find_input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/original_dataset/ActivateBWC_07162021/activatebwc/over_long/"
    default_output_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/original_dataset/ActivateBWC_07162021/activatebwc/"
    
    parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--find_input_dir', type=str, default=default_find_input_dir)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    args = parser.parse_args()

    args.input_sudfolder_list = ['danbin_asr', 'danbin_ori']
    args.output_sudfolder_list = ['danbin_asr_over_long', 'danbin_ori_over_long']
    find_matched_audio()

    # # 剔除声音较小语音数据
    # default_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_05132021/"
    # default_find_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_05132021/small_voice/"
    # default_output_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_05132021/"
    
    # parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    # parser.add_argument('--input_dir', type=str, default=default_input_dir)
    # parser.add_argument('--find_input_dir', type=str, default=default_find_input_dir)
    # parser.add_argument('--output_dir', type=str, default=default_output_dir)
    # args = parser.parse_args()

    # args.input_sudfolder_list = ['xiaoanxiaoan_16k', 'xiaoanxiaoan_8k']
    # args.output_sudfolder_list = ['xiaoanxiaoan_16k_small_voice', 'xiaoanxiaoan_8k_small_voice']
    # find_matched_audio()