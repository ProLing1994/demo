import argparse
import glob
import os
import pandas as pd
import sys
import shutil
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.utils.folder_tools import *

def copy_audio():
    # init 
    file_list = get_sub_filepaths_suffix(args.input_dir, '.wav')
    file_list.sort()

    for idx in tqdm(range(len(file_list))):
        wave_path = file_list[idx]

        # mkdir
        output_subfolder_path = (os.path.dirname(wave_path) + '/').replace(args.input_dir, '')
        output_dir = os.path.join(args.output_dir, output_subfolder_path)
        if not os.path.exists(output_dir):    
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, os.path.basename(wave_path))
        print(wave_path, '->', output_path)
        shutil.copy(wave_path, output_path)
        # shutil.move(wave_path, output_path)


if __name__ == "__main__":
    default_input_dir = "/home/huanyuan/share/huanyuan/audio_data/weakup/weakup_xiaoan8k/平台误唤醒音频/"
    default_output_dir = "/mnt/huanyuan2/data/speech/kws/xiaoan_dataset/original_dataset/Truck_Platformalarm_12022021/truck_platform_alarm_8k/"
    
    parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    args = parser.parse_args()

    copy_audio()