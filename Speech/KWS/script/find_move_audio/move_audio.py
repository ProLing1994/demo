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
    default_input_dir = "/mnt/huanyuan2/model/kws/kws_xiaoan/kws_xiaoan8k_3_2_tc-resnet14-hisi_fbankcpu_kd_11012021/test_straming_wav/实车录制_0427_pytorch/阈值_05_05/其他录音/"
    default_output_dir = "/mnt/huanyuan2/model/kws/kws_xiaoan/kws_xiaoan8k_3_2_tc-resnet14-hisi_fbankcpu_kd_11012021/test_straming_wav/test/"
    
    parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    args = parser.parse_args()

    copy_audio()