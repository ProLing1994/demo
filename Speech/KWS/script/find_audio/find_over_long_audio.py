import argparse
import os
import pandas as pd
import sys
import shutil
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_data_loader_pyimpl import WaveLoader

def find_over_long_audio():
    # init 
    sample_rate = 16000
    wave_list = os.listdir(args.input_dir)

    for idx in tqdm(range(len(wave_list))):
        wave_path = os.path.join(args.input_dir, wave_list[idx])

        # load data
        wave_loader = WaveLoader(sample_rate)
        wave_loader.load_data(wave_path)
        data_length = wave_loader.data_length()

        if data_length > float(args.threshold):
            output_path = os.path.join(args.output_dir, os.path.basename(wave_path).split('.')[0] + '.wav')
            print(wave_path, '->', output_path)
            shutil.copy(wave_path, output_path)


if __name__ == "__main__":
    default_input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/KwsEnglishDataset/activatebwc/"
    default_output_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/special_case/over_long/"
    default_threshold = '2.5'
    
    parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    parser.add_argument('--threshold', type=str, default=default_threshold)
    args = parser.parse_args()

    find_over_long_audio()