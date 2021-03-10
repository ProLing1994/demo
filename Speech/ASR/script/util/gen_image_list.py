import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/ASR')
from impl.asr_data_loader_cimpl import WaveLoader
from impl.asr_feature_cimpl import Feature
from impl.asr_decode_cimpl import Decode


def gen_image_list(args):
    # init 
    sample_rate = 16000
    window_size_ms = 3000
    window_stride_ms = 3000
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
        wave_loader = WaveLoader()
        wave_loader.load_data(wave_path)
        wave_data = wave_loader.to_numpy()

        # sliding window
        windows_times = int((len(wave_data) - window_size_samples) * 1.0 / window_stride_samples) + 1
        for times in range(windows_times):
            print("[Information:] Wave start time: ", times * window_stride_samples)

            audio_data = wave_data[times * int(window_stride_samples): times * int(window_stride_samples) + int(window_size_samples)]

            # cal feature
            # feature = Feature()
            feature = Feature(window_size_samples, int(args.CHW_params.split(",")[2]))
            feature.get_mel_int_feature(audio_data, len(audio_data))
            feature_data = feature.copy_mfsc_feature_int_to()
            # print(np.expand_dims(feature_data, axis=0).shape)
            # print(feature_data)
            # print(feature_data.sum())

            # save img
            feature_data = feature_data.astype(np.uint8)
            assert feature_data.max() <= 255
            assert feature_data.min() >= 0
            output_path = os.path.join(args.output_folder, wave_list[idx].split('.')[0] + '_' + str(times * window_stride_samples) + '.jpg')
            # output_path = os.path.join(args.output_folder, wave_list[idx].split('.')[0] + '_' + str(times * window_stride_samples) + '.png')
            # print(output_path)
            cv2.imwrite(output_path, feature_data)

if __name__ == "__main__":
    # chinese:
    # default_audio_folder = "/home/huanyuan/share/audio_data/第三批数据/安静场景/"
    # default_output_folder = "/home/huanyuan/share/audio_data/第三批数据/安静场景/image_296_64"
    # default_audio_folder = "/home/huanyuan/share/audio_data/第三批数据/闹市场景/"
    # default_output_folder = "/home/huanyuan/share/audio_data/第三批数据/闹市场景/image_296_64"
    # default_CHW_params = "1,296,64"

    # default_audio_folder = "/home/huanyuan/share/audio_data/第三批数据/安静场景/"
    # default_output_folder = "/home/huanyuan/share/audio_data/第三批数据/安静场景/image_296_56"
    # default_audio_folder = "/home/huanyuan/share/audio_data/第三批数据/闹市场景/"
    # default_output_folder = "/home/huanyuan/share/audio_data/第三批数据/闹市场景/image_296_56"
    default_audio_folder = "/home/huanyuan/share/audio_data/"
    default_output_folder = "/home/huanyuan/share/audio_data/image_296_56"
    default_CHW_params = "1,296,56"

    # # english:
    # default_audio_folder = "/home/huanyuan/share/audio_data/english_wav/"
    # default_output_folder = "/home/huanyuan/share/audio_data/english_wav/image_296_64"
    # default_CHW_params = "1,296,64"

    parser = argparse.ArgumentParser(description='Streamax ASR Demo Engine')
    parser.add_argument('-i', '--audio_folder', type=str, default=default_audio_folder)
    parser.add_argument('-o', '--output_folder', type=str, default=default_output_folder)
    parser.add_argument('--CHW_params', type=str, default=default_CHW_params)
    args = parser.parse_args()

    gen_image_list(args)