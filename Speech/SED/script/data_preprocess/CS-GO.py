# 注：请执行操作顺序：
# 1. 执行本文件，根据 txt 文件和 wav 文件自动截取音频，保存截取后的音频（该 txt 文件来自音频裁剪工具）

import argparse
import glob
import librosa
import os 
import wave
import yaml

from tqdm import tqdm

ignore_list = ['boltpul', 'boltrelease', 'boltback', 'boltforward', 'bknife', 'boxin', 'boxout', 'chain', 'cliphit', 'clipin', 
                'clipout', 'coverdown', 'coverup', 'cliprelease', 'cliptouch', 'c4', 'decoy', 'draw', 'fire_', 'grenade', 'hammer', 
                'he_bounce', 'hegrenade', 'insertshell', 'knife', 'lookat', 'molotov', 'off', 'on', 'pump', 'pump_back', 'pump_forward', 
                'pinpull', 'prepare', 'reloadstar', 'sg_', 'smoke', 'sliderelease', 'stilletto', 'slideback', 'sliderelease', 'slideforward', 
                'siderelease', 'sideback', 'silencer', 'sensor', 'taser', 'taunt_tap', 'ursus', 'widow', 'zoom', 'zoom_in', 'zoom_out']

def main():
    # mkdir 
    if not os.path.exists(os.path.join(args.output_folder, args.label)):
        os.makedirs(os.path.join(args.output_folder, args.label))

    # load audio_list 
    audio_list = []
    audio_list = glob.glob(os.path.join(args.input_folder, '*/*' + args.audio_suffix))
    audio_list.sort()

    for idx in tqdm(range(len(audio_list))):
        wave_path = audio_list[idx]

        ignore_wave_bool = False
        for ignore_idx in range(len(ignore_list)):
            if ignore_list[ignore_idx] in os.path.basename(wave_path):
                ignore_wave_bool = True
        
        if ignore_wave_bool:
            continue

        # output 
        output_path = os.path.join(args.output_folder, args.label, args.dataset_name + args.label + "_{}".format(os.path.basename(wave_path)))
        os.system("cp {} {}".format(wave_path, output_path))


if __name__ == "__main__":
    default_input_folder = "/mnt/huanyuan/data/speech/sed/CS-GO/original_dataset/weapons/"
    default_output_folder = "/mnt/huanyuan/data/speech/sed/CS-GO/processed_dataset/"
    default_dataset_name = "CSGO_SED_"
    default_label = "gunshot"

    parser = argparse.ArgumentParser(description='CS-GO Engine')
    parser.add_argument('--input_folder', type=str, default=default_input_folder)
    parser.add_argument('--output_folder', type=str, default=default_output_folder)
    parser.add_argument('--dataset_name', type=str, default=default_dataset_name)
    parser.add_argument('--label', type=str, default=default_label)
    parser.add_argument('--audio_suffix', type=str, default=".wav")
    args = parser.parse_args()

    main()