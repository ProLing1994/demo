import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_data_loader_pyimpl import WaveLoader
from ASR.impl.asr_feature_pyimpl import Feature
# from ASR.impl.asr_feature_cimpl import Feature
import ASR.impl.asr_decode_cimpl as Decode_C
import ASR.impl.asr_decode_pyimpl as Decode_Python

caffe_root = "/home/huanyuan/code/caffe/"
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')
import caffe

def model_init(prototxt, model, net_input_name, CHW_params, use_gpu=False):
    if use_gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
        print("[Information:] GPU mode")
    else:
        caffe.set_mode_cpu()
        print("[Information:] CPU mode")
    net = caffe.Net(prototxt, model, caffe.TEST)
    net.blobs[net_input_name].reshape(1, int(CHW_params[0]), int(CHW_params[1]), int(CHW_params[2])) 
    return net


def asr_model_test(args):
    # init 
    sample_rate = args.sample_rate
    window_size_ms = args.window_size_ms
    window_stride_ms = args.window_stride_ms
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    # model init
    net = model_init(args.prototxt_path, args.model_path, args.net_input_name, args.chw_params.split(","), args.gpu)

    # load wave
    wave_loader = WaveLoader(sample_rate)
    wave_loader.load_data(args.audio_wav)
    wave_data = wave_loader.to_numpy()
    print("[Information:] Audio path: ", args.audio_wav)

    # sliding window
    windows_times = int((len(wave_data) - window_size_samples) * 1.0 / window_stride_samples) + 1
    for times in range(windows_times):
        print("[Information:] Wave start time: ", times * window_stride_samples)

        audio_data = wave_data[times * int(window_stride_samples): times * int(window_stride_samples) + int(window_size_samples)]
        
        # cal feature
        feature = Feature(sample_rate, window_size_samples/sample_rate, args.feature_freq, args.nfilt)
        feature.get_mel_int_feature(audio_data, len(audio_data))
        feature_data = feature.copy_mfsc_feature_int_to()
        feature_data = feature_data.astype(np.uint8)
        feature_data = feature_data[: args.feature_time, : args.feature_freq]
        if args.transpose:
            feature_data = feature_data.T
        print(feature_data.shape)

        # forward
        net.blobs[args.net_input_name].data[...] = np.expand_dims(feature_data, axis=0)

        net_output = net.forward()[args.net_output_name]
        net_output = np.squeeze(net_output)
        net_output = net_output.T
        print(net_output.shape)
        print(net_output)
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Streamax ASR Demo Engine')
    args = parser.parse_args()

    # # kws weakup: xiaoan8k
    # args.model_path = "/mnt/huanyuan/model/audio_model/hisi_model/kws_xiaoan8k_tc_resnet14/kws_xiaoan8k_tc_resnet14_hisi_3_1_05272021.caffemodel"
    # args.prototxt_path = "/mnt/huanyuan/model/audio_model/hisi_model/kws_xiaoan8k_tc_resnet14/kws_xiaoan8k_tc_resnet14_hisi_3_1_05272021.prototxt"
    # args.chw_params = "1,48,144"
    # args.net_input_name = "data"
    # args.net_output_name = "prob"
    # args.transpose = True
    # args.nfilt = 48
    # args.feature_freq = 48
    # args.feature_time = 144
    # args.gpu = True
    # args.audio_wav = "/home/huanyuan/share/audio_data/weakup_xiaoan8k/test/xiaoan8k_1_1_04082021_validation_60.wav"

    # kws weakup: xiaorui16k
    args.model_path = "/mnt/huanyuan/model/audio_model/hisi_model/kws_xiaorui16k_tc_resnet14/kws_xiaorui8k_tc_resnet14_hisi_6_1_05272021.caffemodel"
    args.prototxt_path = "/mnt/huanyuan/model/audio_model/hisi_model/kws_xiaorui16k_tc_resnet14/kws_xiaorui8k_tc_resnet14_hisi_6_1_05272021.prototxt"
    args.chw_params = "1,64,192"
    args.net_input_name = "data"
    args.net_output_name = "prob"
    args.transpose = True
    args.nfilt = 64
    args.feature_freq = 64
    args.feature_time = 192
    args.gpu = True
    args.audio_wav = "/mnt/huanyuan/model/test_straming_wav/xiaorui_1_4_04302021_validation_60.wav"

    # # 16k & 4s
    # args.sample_rate = 16000
    # args.window_size_ms = 4000
    # args.window_stride_ms = 3000
    
    # # 16k & 3s
    # args.sample_rate = 16000
    # args.window_size_ms = 3000
    # args.window_stride_ms = 3000

    # 16k & 2s
    args.sample_rate = 16000
    args.window_size_ms = 2000
    args.window_stride_ms = 2000

    # # 8k & 1.5s
    # args.sample_rate = 8000
    # args.window_size_ms = 1500
    # args.window_stride_ms = 1500

    asr_model_test(args)