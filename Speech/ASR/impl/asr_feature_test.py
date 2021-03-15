import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/ASR')
# from impl.asr_data_loader_cimpl import WaveLoader
from impl.asr_data_loader_pyimpl import WaveLoader
import impl.asr_feature_cimpl
import impl.asr_feature_pyimpl


def ase_feature_test(args):
    # init 
    sample_rate = 16000
    window_size_ms = 2000
    window_stride_ms = 2000
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    wave_list = os.listdir(args.audio_folder)
    wave_list.sort()
    for idx in range(len(wave_list)):
        
        if not wave_list[idx].endswith('.wav'):
            continue

        wave_path = os.path.join(args.audio_folder, wave_list[idx])
        print("[Information:] Audio path: ", wave_path)
        
        # load wave
        wave_loader = WaveLoader()
        wave_loader.load_data(wave_path)
        wave_data = wave_loader.to_numpy()

        # sliding window
        windows_times = int((len(wave_data) - window_size_samples) * 1.0 / window_stride_samples) + 1
        for times in range(windows_times):
            print("[Information:] Wave start time: ", times * window_stride_samples)

            audio_data = wave_data[times * int(window_stride_samples): times * int(window_stride_samples) + int(window_size_samples)]
            print(audio_data.shape)
            print(audio_data[:10])

            # cal feature
            # cimpl
            feature_cimpl = impl.asr_feature_cimpl.Feature(window_size_samples, int(args.CHW_params.split(",")[2]))
            feature_cimpl.get_mel_int_feature(audio_data, len(audio_data))
            feature_data_cimpl = feature_cimpl.copy_mfsc_feature_int_to()
            print(np.expand_dims(feature_data_cimpl, axis=0).shape)
            print(feature_data_cimpl)

            # check feature data
            feature_data_cimpl = feature_data_cimpl.astype(np.uint8)
            assert feature_data_cimpl.max() <= 255
            assert feature_data_cimpl.min() >= 0

            # pyimpl
            feature_pyimpl = impl.asr_feature_pyimpl.Feature(window_size_samples, int(args.CHW_params.split(",")[2]))
            feature_pyimpl.get_mel_int_feature(audio_data, len(audio_data))
            feature_data_pyimpl = feature_pyimpl.copy_mfsc_feature_int_to()
            print(np.expand_dims(feature_data_pyimpl, axis=0).shape)
            print(feature_data_pyimpl)

            # check feature data
            feature_data_pyimpl = feature_data_pyimpl.astype(np.uint8)
            assert feature_data_pyimpl.max() <= 255
            assert feature_data_pyimpl.min() >= 0

            print("cimpl == pympl:", (feature_data_cimpl == feature_data_pyimpl).all())

if __name__ == "__main__":
    # chinese:
    # default_audio_folder = "/home/huanyuan/share/audio_data/第三批数据/安静场景/"
    # default_audio_folder = "/home/huanyuan/share/audio_data/第三批数据/闹市场景/"
    default_audio_folder = "/home/huanyuan/share/audio_data/"
    default_CHW_params = "1,296,48"
    # default_CHW_params = "1,196,64"

    # # english:
    # default_audio_folder = "/home/huanyuan/share/audio_data/english_wav/"
    # default_CHW_params = "1,296,64"

    parser = argparse.ArgumentParser(description='Streamax ASR Demo Engine')
    parser.add_argument('-i', '--audio_folder', type=str, default=default_audio_folder)
    parser.add_argument('--CHW_params', type=str, default=default_CHW_params)
    args = parser.parse_args()

    ase_feature_test(args)