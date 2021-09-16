import argparse
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
import ASR.impl.asr_data_loader_pyimpl as WaveLoader_Python
from KWS.utils.folder_tools import *

def wav_merge(args):
    # init 
    wave_loader = WaveLoader_Python.WaveLoader_Librosa(args.sample_rate)
    wave_loader = WaveLoader_Python.WaveLoader_Soundfile(args.sample_rate)

    file_list = get_sub_filepaths_suffix(args.input_dir, '.wav')
    file_dict = { int(os.path.splitext(os.path.basename(file_idx))[0].split("_")[-1]) : file_idx for file_idx in file_list }
    file_dict = { file_idx : file_dict[file_idx] for file_idx in sorted(file_dict) } 
    wave_data_list = []

    for idx in tqdm(range(len(file_dict))):
        wave_path = file_dict[idx]
        if str(os.path.basename(wave_path)).startswith(args.prefix):
            print("path: ", wave_path) 
            wave_loader.load_data(wave_path)
            wave_data = wave_loader.to_numpy()
            wave_data_list.extend(list(wave_data))

    output_path = os.path.join(args.output_dir, args.prefix + "merge.wav")
    wave_loader.save_data(np.array(wave_data_list), output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    args = parser.parse_args()
    args.input_dir = "/home/huanyuan/temp/wave_2/"
    args.output_dir = "/home/huanyuan/temp/"
    args.prefix = "demo_output_"
    args.sample_rate = 16000
    wav_merge(args)