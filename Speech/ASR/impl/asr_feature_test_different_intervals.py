import argparse
import numpy as np
import os
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_data_loader_pyimpl import WaveLoader
from ASR.impl.asr_feature_pyimpl import Feature


def cal_feature_1s(wave_data, times):
    audio_data = wave_data[times * int(window_stride_samples): times * int(window_stride_samples) + int(window_size_samples)]
    # print(audio_data)
    feature = Feature(window_size_samples, int(feature_freq))
    feature.get_mel_int_feature(audio_data, len(audio_data))
    feature_data = feature.copy_mfsc_feature_int_to()
    return feature_data

def cal_feature_2s(wave_data, times):
    audio_data = wave_data[times * int(window_stride_samples): times * int(window_stride_samples) + int(window_size_samples) * 2]
    # print(audio_data)
    feature = Feature(window_size_samples * 2, int(feature_freq))
    feature.get_mel_int_feature(audio_data, len(audio_data))
    feature_data = feature.copy_mfsc_feature_int_to()
    return feature_data

def cal_feature_retain(wave_data, times):
    audio_data = wave_data[times * int(window_stride_samples) - (int(window_size_samples_retain) - int(window_size_samples)): times * int(window_stride_samples) + int(window_size_samples)]
    # print(audio_data)
    feature = Feature(window_size_samples_retain, int(feature_freq))
    feature.get_mel_int_feature(audio_data, len(audio_data))
    feature_data = feature.copy_mfsc_feature_int_to()
    return feature_data

def KWS_ASR_offine():
    # load wave
    wave_loader = WaveLoader(sample_rate)
    wave_loader.load_data(args.input_wav)
    wave_data = wave_loader.to_numpy()

    # 计算 1s 特征
    times = 10
    feature_data_10_1s = cal_feature_1s(wave_data, times)
    print("feature shape: ", feature_data_10_1s.shape)

    # 计算 1s 特征
    times = 11
    feature_data_11_1s = cal_feature_1s(wave_data, times)
    print("feature shape: ", feature_data_11_1s.shape)

    # 计算 2s 特征
    times = 10
    feature_data = cal_feature_2s(wave_data, times)
    print("feature shape: ", feature_data.shape)
    for idx in range(feature_data_10_1s.shape[0]):
        if not (feature_data_10_1s[idx,:] == feature_data[idx,:]).all():
            print("different feature", idx)

    for idx in range(feature_data_11_1s.shape[0]):
        if not (feature_data_11_1s[idx,:] == feature_data[-feature_data_11_1s.shape[0] + idx,:]).all():
            print("different feature", idx)

    print("feature_data_10_1s: ")
    print(feature_data_10_1s[95, :])

    print("feature_data: ")
    print(feature_data[95, :])
    print(feature_data[96, :])
    print(feature_data[97, :])
    print(feature_data[98, :])
    print(feature_data[99, :])
    print(feature_data[100, :])

    print("feature_data_11_1s: ")
    print(feature_data_11_1s[0, :])

    # 保留一定时长音频数据，计算并补齐特征
    times = 11
    feature_data_retain = cal_feature_retain(wave_data, times)
    print("feature shape: ", feature_data_retain.shape)
    for idx in range(feature_data_retain.shape[0]):
        if not (feature_data_retain[idx,:] == feature_data[-feature_data_retain.shape[0] + idx,:]).all():
            print("different feature", idx)

    print(feature_data_10_1s[0:90 ,:].shape)
    print(np.concatenate((feature_data_10_1s[0: 90 , :], feature_data_retain), axis=0).shape)
    feature_merge = np.concatenate((feature_data_10_1s[0: 90 , :], feature_data_retain), axis=0)
    print("feature_merge == feature_data: ", (feature_merge == feature_data).all())


if __name__ == "__main__":0
    # params
    sample_rate = 16000
    window_size_ms = 1000                   # 每次送入 1s 数据
    window_stride_ms = 1000                 # 每次间隔 1s 时间
    window_size_ms_retain = 1100

    feature_freq = 64
    kws_feature_time = 196
    kws_stride_feature_time = 10            # kws 每间隔 10 个 feature_time 进行一次检索
    kws_detection_threshold = 0.8           # kws 检测阈值 0.8
    kws_detection_number_threshold = 0.5   # kws 计数阈值 0.5

    # init 
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    window_size_samples_retain = int(sample_rate * window_size_ms_retain / 1000)

    # argparse
    default_input_wav = "/mnt/huanyuan/model/test_straming_wav/xiaorui_12162020_training_60_001.wav"
    default_output_folder = "/mnt/huanyuan/model/audio_model/caffe_model/kws_xiaorui_res15/"

    parser = argparse.ArgumentParser(description='Streamax KWS ASR offine Engine')
    parser.add_argument('--input_wav', type=str, default=default_input_wav)
    parser.add_argument('--output_folder', type=str, default=default_output_folder)
    args = parser.parse_args()

    KWS_ASR_offine()