import argparse
import os
import pandas as pd
import sys
import shutil
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_data_loader_pyimpl import WaveLoader

def find_audio():
    # mkdir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # init 
    wave_list = os.listdir(args.input_dir)

    for idx in tqdm(range(len(wave_list))):
        wave_path = os.path.join(args.input_dir, wave_list[idx])
        find_path = os.path.join(args.find_input_dir, wave_list[idx])

        if not os.path.exists(find_path):
            output_path = os.path.join(args.output_dir, os.path.basename(wave_path).split('.')[0] + '.wav')
            print(wave_path, '->', output_path)
            shutil.copy(wave_path, output_path)


if __name__ == "__main__":
    default_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_05132021/xiaoan_0422/error/mic/"
    default_find_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_05132021/xiaoan_0422/error/kaldi_cut_keyword/mic/"
    default_output_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_05132021/xiaoan_0422/error/kaldi_cut_keyword/unuse/"
    
    parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--find_input_dir', type=str, default=default_find_input_dir)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    args = parser.parse_args()

    find_audio()