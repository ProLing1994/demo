# 注：请执行操作顺序：
# 1. 执行本文件，根据 txt 文件和 wav 文件自动截取音频，保存截取后的音频（该 txt 文件来自音频裁剪工具）

import argparse
import glob
import librosa
import os 
import pandas as pd
import wave
import yaml

from tqdm import tqdm

def main():
    # mkdir 
    if not os.path.exists(os.path.join(args.output_folder, args.label)):
        os.makedirs(os.path.join(args.output_folder, args.label))

    # load audio_list 
    audio_dict_s = {}
    audio_dict_c = {}
    file_list = []
    audio_list = []
    audio_list = glob.glob(os.path.join(args.input_folder, '*/*/*' + args.audio_suffix))
    audio_list.sort()

    for idx in tqdm(range(len(audio_list))):
        wave_path = audio_list[idx]

        if 'Shot ' not in wave_path and 'Shots ' not in wave_path and 'shot ' not in wave_path and 'shots ' not in wave_path:
            continue

        # output 
        folder_name = os.path.basename(os.path.dirname(wave_path))
        folder_name = "".join(folder_name.split('&'))
        folder_name = "".join(folder_name.split(' '))
        folder_name = "".join(folder_name.split('-'))

        if 'Shot ' in wave_path or 'shot ' in wave_path:
            if folder_name not in audio_dict_s:
                audio_dict_s[folder_name] = 1
            file_name = "{}_{:0>3d}{}".format('Single', audio_dict_s[folder_name], args.audio_suffix)
            audio_dict_s[folder_name] += 1

        if 'Shots ' in wave_path or 'shots ' in wave_path:
            if folder_name not in audio_dict_c:
                audio_dict_c[folder_name] = 1
            file_name = "{}_{:0>3d}{}".format('Auto', audio_dict_c[folder_name], args.audio_suffix)
            audio_dict_c[folder_name] += 1

        output_path = os.path.join(args.output_folder, args.label, args.dataset_name + args.label + "_{}_{}".format(folder_name, file_name))
        output_path = output_path.replace('+', '')
        os.system("cp '{}' {}".format(wave_path, output_path))
        file_list.append({"input path": wave_path, "output": output_path})

    file_pd = pd.DataFrame(file_list)
    file_pd.to_csv(os.path.join(args.output_folder, args.label, args.label + '.csv'), index=False, encoding="utf_8_sig")


if __name__ == "__main__":
    default_input_folder = "/mnt/huanyuan/data/speech/sed/THE_GUNS_BUNDLE/original_dataset/THE_GUNS_BUNDLE/"
    default_output_folder = "/mnt/huanyuan/data/speech/sed/THE_GUNS_BUNDLE/processed_dataset/"
    default_dataset_name = "BUNSBUNDLE_SED_"
    default_label = "gunshot"

    parser = argparse.ArgumentParser(description='BUNS-BUNDLE Engine')
    parser.add_argument('--input_folder', type=str, default=default_input_folder)
    parser.add_argument('--output_folder', type=str, default=default_output_folder)
    parser.add_argument('--dataset_name', type=str, default=default_dataset_name)
    parser.add_argument('--label', type=str, default=default_label)
    parser.add_argument('--audio_suffix', type=str, default=".wav")
    args = parser.parse_args()

    main()