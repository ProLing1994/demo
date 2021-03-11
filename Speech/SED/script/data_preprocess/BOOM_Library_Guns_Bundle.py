# 注：请执行操作顺序：
# 1. 执行本文件，根据 wav 文件及 VAD 工具自动截取音频，保存截取后的音频

import argparse
from auditok import split
import glob
import librosa
import os 
import pandas as pd
import wave
import yaml

from tqdm import tqdm

ignore_list = ['Handling', 'Loading']
gun_types_list = ['Beretta M9', 'Colt1911', 'FNH FNP40', 'FNH FNP45', 'KelTec P318', 'S&W Model 66', 'S&W Model66', 'SKS M59', 'AK47', 'M4',
                    'Dragunov PSL', 'GSG5', 'Mosin Nagant', 'MosinNagant', 'Ruger 1022', 'Ithica M37', 'IthicaM37', 'Maverick 88', 'Maverick88', 'Maverick 88 FM',
                    'Maverick88FM', 'Winchester1300']

def vad_save(args, wave_path, audio_dict, gun_type, file_list, status='Auto'):
    # VAD
    tmep_path = os.path.join(os.path.dirname(os.path.dirname(args.output_folder)), 'temp.wav')
    os.system("sox '{}' -c 1 -b 16 -r 48000 -e signed-integer '{}'".format(wave_path, tmep_path))
    audio_regions = split(tmep_path, 0.01, 10, 1.0, False, False, energy_threshold=60)

    for region in audio_regions:
        file_name = "{}_{:0>3d}{}".format(status, audio_dict[gun_type], args.audio_suffix)
        audio_dict[gun_type] += 1

        output_path = os.path.join(args.output_folder, args.label, args.dataset_name + args.label + "_{}_{}".format(gun_type, file_name))
        filename = region.save(output_path)

        file_list.append({"input path": wave_path, "output": output_path})
    return file_list

def main():
    # mkdir 
    if not os.path.exists(os.path.join(args.output_folder, args.label)):
        os.makedirs(os.path.join(args.output_folder, args.label))

    # load audio_list 
    audio_dict_single = {}
    audio_dict_burst = {}
    audio_dict_auto = {}
    file_list = []
    audio_list = []
    audio_list = glob.glob(os.path.join(args.input_folder, '*/*/*' + args.audio_suffix))
    audio_list += glob.glob(os.path.join(args.input_folder, '*' + args.audio_suffix))
    audio_list.sort()

    for wave_idx in tqdm(range(len(audio_list))):
        wave_path = audio_list[wave_idx]

        ignore_wave_bool = False
        for ignore_idx in range(len(ignore_list)):
            if ignore_list[ignore_idx] in os.path.basename(wave_path):
                ignore_wave_bool = True
        
        if ignore_wave_bool:
            continue
        
        find_gun_type_bool = False
        for type_idx in range(len(gun_types_list)):
            gun_type = gun_types_list[type_idx]

            if gun_type in wave_path:
                find_gun_type_bool = True

                # gun_type = os.path.basename(os.path.dirname(wave_path))
                gun_type = "".join(gun_type.split(' '))
                gun_type = "".join(gun_type.split('-'))
                gun_type = "".join(gun_type.split('&'))
                if 'Auto' in wave_path:
                    if gun_type not in audio_dict_auto:
                        audio_dict_auto[gun_type] = 1

                    file_list = vad_save(args, wave_path, audio_dict_auto, gun_type, file_list, 'Auto')

                elif 'Bursts' in wave_path or 'Double Tap' in wave_path:
                    if gun_type not in audio_dict_burst:
                        audio_dict_burst[gun_type] = 1

                    file_list = vad_save(args, wave_path, audio_dict_burst, gun_type, file_list, 'Burst')

                elif 'Single Shots' in wave_path or 'Shot Sequence' in wave_path or 'SingleShots' in wave_path:
                    if gun_type not in audio_dict_single:
                        audio_dict_single[gun_type] = 1

                    file_list = vad_save(args, wave_path, audio_dict_single, gun_type, file_list, 'Single')
                    
                else:
                    print(wave_path)

                break

        if not find_gun_type_bool:
            print(wave_path)
                
    file_pd = pd.DataFrame(file_list)
    file_pd.to_csv(os.path.join(args.output_folder, args.label, args.label + '_{}.csv'.format(args.csv_num)), index=False, encoding="utf_8_sig")
    

if __name__ == "__main__":
    # default_input_folder = "/mnt/huanyuan/data/speech/sed/BOOM_Library/original_dataset/11_Guns_Bundle/Guns - Construction Kit/"
    # default_output_folder = "/mnt/huanyuan/data/speech/sed/BOOM_Library/processed_dataset/Guns/"
    # default_dataset_name = "BOOMLibrary11_SED_"
    # default_label = "gunshot"
    # default_csv_num = 0

    default_input_folder = "/mnt/huanyuan/data/speech/sed/BOOM_Library/original_dataset/11_Guns_Bundle/Guns - Designed/"
    default_output_folder = "/mnt/huanyuan/data/speech/sed/BOOM_Library/processed_dataset/Guns/"
    default_dataset_name = "BOOMLibrary11_SED_"
    default_label = "gunshot_designed"
    default_csv_num = 1

    parser = argparse.ArgumentParser(description='BOOM_Library Engine')
    parser.add_argument('--input_folder', type=str, default=default_input_folder)
    parser.add_argument('--output_folder', type=str, default=default_output_folder)
    parser.add_argument('--dataset_name', type=str, default=default_dataset_name)
    parser.add_argument('--label', type=str, default=default_label)
    parser.add_argument('--csv_num', type=str, default=default_csv_num)
    parser.add_argument('--audio_suffix', type=str, default=".wav")
    args = parser.parse_args()

    main()