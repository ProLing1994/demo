import argparse
import glob
import os
import pandas as pd
import sys
import shutil
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_data_loader_pyimpl import WaveLoader

def move_audio():
    # init 
    file_list = glob.glob(os.path.join(args.input_dir, '*' + ".wav"))
    file_list += glob.glob(os.path.join(args.input_dir, '*/*' + ".wav"))
    file_list += glob.glob(os.path.join(args.input_dir, '*/*/*' + ".wav"))
    file_list += glob.glob(os.path.join(args.input_dir, '*/*/*/*' + ".wav"))
    file_list += glob.glob(os.path.join(args.input_dir, '*/*/*/*/*' + ".wav"))
    file_list.sort()

    for idx in tqdm(range(len(file_list))):
        wave_path = file_list[idx]

        # mkdir
        output_subfolder_path = (os.path.dirname(wave_path) + '/').replace(args.input_dir, '')
        # output_dir = os.path.join(args.output_dir, args.output_folder_name + '_' + output_subfolder_path)
        output_dir = os.path.join(args.output_dir, args.output_folder_name, output_subfolder_path)
        if not os.path.exists(output_dir):    
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, os.path.basename(wave_path))
        print(wave_path, '->', output_path)
        shutil.copy(wave_path, output_path)


if __name__ == "__main__":
    default_input_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_activatebwc_1_5_res15_fbankcpu_03222021/test_straming_wav/ADkit_weiboyulu/"
    default_output_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/original_dataset/Adkit_record_03312021/adkit_record/"
    default_output_folder_name = "ADkit_weiboyulu"
    
    parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    parser.add_argument('--output_folder_name', type=str, default=default_output_folder_name)
    args = parser.parse_args()

    move_audio()