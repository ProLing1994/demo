import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/ASR')
from impl.asr_data_loader_pyimpl import WaveLoader
from impl.asr_feature_cimpl import Feature
# from impl.asr_feature_pyimpl import Feature
from impl.asr_decode_cimpl import Decode


def gen_image_list(args):
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

            audio_data_list = audio_data.tolist()
            while len(audio_data_list) < window_size_samples:
                audio_data_list.append(0.0)
            audio_data = np.array(audio_data_list)
            
            # cal feature
            feature = Feature(sample_rate, window_size_samples/sample_rate, int(args.CHW_params.split(",")[2]), args.nfilt)
            feature.get_mel_int_feature(audio_data, len(audio_data))
            feature_data = feature.copy_mfsc_feature_int_to()
            print(np.expand_dims(feature_data, axis=0).shape)
            # print(feature_data)
            # print(feature_data.sum())

            # save img
            feature_data = feature_data.astype(np.uint8)
            assert feature_data.max() <= 255
            assert feature_data.min() >= 0
            output_path = os.path.join(args.output_folder, wave_list[idx].split('.')[0] + '_' + str(times * window_stride_samples) + '.jpg')
            
            if args.transpose:
                feature_data = feature_data.T
            cv2.imwrite(output_path, feature_data)

if __name__ == "__main__":
    # # asr mandarin:
    # # default_audio_folder = "/home/huanyuan/share/audio_data/mandarin_wav/安静场景/"
    # # default_output_folder = "/home/huanyuan/share/audio_data/mandarin_wav/安静场景/image_396_64"
    # default_audio_folder = "/home/huanyuan/share/audio_data/mandarin_wav/闹市场景/"
    # default_output_folder = "/home/huanyuan/share/audio_data/mandarin_wav/闹市场景/image_396_64"
    # default_CHW_params = "1,396,64"
    # default_transpose = False
    
    # default_audio_folder = "/home/huanyuan/share/audio_data/mandarin_wav/安静场景/"
    # default_output_folder = "/home/huanyuan/share/audio_data/mandarin_wav/安静场景/image_296_56"
    # default_audio_folder = "/home/huanyuan/share/audio_data/mandarin_wav/闹市场景/"
    # default_output_folder = "/home/huanyuan/share/audio_data/mandarin_wav/闹市场景/image_296_56"
    # default_audio_folder = "/home/huanyuan/share/audio_data/"
    # default_output_folder = "/home/huanyuan/share/audio_data/image_296_56"
    # default_CHW_params = "1,296,56"
    # default_transpose = False

    # # kws: xiaorui
    # default_audio_folder = "/home/huanyuan/share/audio_data/weakup_xiaorui/xiaorui/"
    # default_audio_folder = "/home/huanyuan/share/audio_data/weakup_xiaorui/xiaorui_long/"
    # default_audio_folder = "/home/huanyuan/share/audio_data/weakup_xiaorui/other/"
    # default_output_folder = "/home/huanyuan/share/audio_data/weakup_xiaorui/image_196_64"
    # default_CHW_params = "1,196,64"
    # default_transpose = False
    # # default_output_folder = "/home/huanyuan/share/audio_data/weakup_xiaorui/image_64_196"
    # # default_CHW_params = "1,196,64"
    # # default_transpose = True

    # # kws: activatebwc
    # default_audio_folder = "/home/huanyuan/share/audio_data/weakup_activatebwc/activatebwc/"
    # # default_audio_folder = "/home/huanyuan/share/audio_data/weakup_activatebwc/activatebwc_long/"
    # # default_audio_folder = "/home/huanyuan/share/audio_data/weakup_activatebwc/other/"
    # # default_output_folder = "/home/huanyuan/share/audio_data/weakup_activatebwc/image_64_196"
    # default_output_folder = "/home/huanyuan/share/audio_data/weakup_activatebwc/temp/"
    # default_CHW_params = "1,196,64"
    # default_transpose = True

    # # asr english:
    # default_audio_folder = "/home/huanyuan/share/audio_data/english_wav/"
    # default_output_folder = "/home/huanyuan/share/audio_data/english_wav/image_296_64"
    # default_CHW_params = "1,296,64"
    # default_transpose = False

    # kws: xiaoan8k
    default_audio_folder = "/home/huanyuan/share/audio_data/weakup_xiaoan8k/xiaoan8k/"
    # default_audio_folder = "/home/huanyuan/share/audio_data/weakup_xiaoan8k/xiaoan8k_long/"
    # default_audio_folder = "/home/huanyuan/share/audio_data/weakup_xiaoan8k/other/"
    default_output_folder = "/home/huanyuan/share/audio_data/weakup_xiaoan8k/image_48_146/"
    default_CHW_params = "1,146,48"
    default_transpose = True

    # # weakup & asr:
    # default_audio_folder = "/home/huanyuan/share/audio_data/kws_weakup_asr/test/"
    # default_output_folder = "/home/huanyuan/share/audio_data/kws_weakup_asr/test/"
    # default_CHW_params = "1,296,64"
    # default_transpose = False

    parser = argparse.ArgumentParser(description='Streamax ASR Demo Engine')
    parser.add_argument('-i', '--audio_folder', type=str, default=default_audio_folder)
    parser.add_argument('-o', '--output_folder', type=str, default=default_output_folder)
    parser.add_argument('--CHW_params', type=str, default=default_CHW_params)
    parser.add_argument('--transpose', action='store_true', default=default_transpose)
    args = parser.parse_args()

    # # 16k & 4s
    # args.sample_rate = 16000
    # args.window_size_ms = 4000
    # args.window_stride_ms = 2000
    # args.nfilt = 64

    # # 16k & 3s
    # args.sample_rate = 16000
    # args.window_size_ms = 3000
    # args.window_stride_ms = 2000
    # args.nfilt = 64

    # # 16k & 2s
    # args.sample_rate = 16000
    # args.window_size_ms = 2000
    # args.window_stride_ms = 2000
    # args.nfilt = 64

    # 8k & 1.5s
    args.sample_rate = 8000
    args.window_size_ms = 1500
    args.window_stride_ms = 1500
    args.nfilt = 48
    gen_image_list(args)