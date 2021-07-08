import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/ASR')
from impl.asr_data_loader_pyimpl import WaveLoader_Librosa as WaveLoader
from impl.asr_feature_cimpl import Feature
# from impl.asr_feature_pyimpl import Feature
from impl.asr_decode_cimpl import Decode


def gen_wav_list(args):
    # init 
    sample_rate = args.sample_rate
    window_size_ms = args.window_size_ms
    window_stride_ms = args.window_stride_ms
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    # mkdir 
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    wave_list = os.listdir(args.audio_folder)
    wave_list.sort()
    for idx in range(len(wave_list)):
        
        if not wave_list[idx].endswith('.wav'):
            continue

        wave_path = os.path.join(args.audio_folder, wave_list[idx])
        print("[Information:] Audio path: ", wave_path)
        
        # load wave
        wave_loader = WaveLoader(sample_rate)
        wave_loader.load_data(wave_path)
        wave_data = wave_loader.to_numpy()

        # sliding window
        windows_times = int((len(wave_data) - window_size_samples) * 1.0 / window_stride_samples) + 1
        for times in range(windows_times):
            print("[Information:] Wave start time: ", times * window_stride_samples)

            audio_data = wave_data[times * int(window_stride_samples): times * int(window_stride_samples) + int(window_size_samples)]

            # save wav
            output_path = os.path.join(args.output_folder, wave_list[idx].split('.')[0] + '_' + str(times) + '.wav')
            wave_loader.save_data(np.array(audio_data), output_path)


if __name__ == "__main__":
    default_audio_folder = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/test_dataset/非常珍贵的外籍人士专门录制的语料/"
    default_output_folder = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/test_dataset/非常珍贵的外籍人士专门录制的语料/wav_list/"

    parser = argparse.ArgumentParser(description='Streamax ASR Demo Engine')
    args = parser.parse_args()
    args.audio_folder = default_audio_folder
    args.output_folder = default_output_folder
    
    # 16k & 4s
    args.sample_rate = 16000
    args.window_size_ms = 60000
    args.window_stride_ms = 60000
    gen_wav_list(args)