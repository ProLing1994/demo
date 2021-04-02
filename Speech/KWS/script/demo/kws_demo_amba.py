import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/ASR')
# from impl.asr_data_loader_cimpl import WaveLoader
# from impl.asr_feature_cimpl import Feature
from impl.asr_data_loader_pyimpl import WaveLoader
from impl.asr_feature_pyimpl import Feature

caffe_root = "/home/huanyuan/code/caffe_ssd/"
sys.path.insert(0, caffe_root+'python')
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


def decode_sliding_window(args, wave_data, window_size_samples, window_stride_samples):
    # sliding window
    windows_times = int((len(wave_data) - window_size_samples) * 1.0 / window_stride_samples) + 1
    for times in range(windows_times):
        print("[Information:] Wave start time: ", times * window_stride_samples)

        audio_data = wave_data[times * int(window_stride_samples): times * int(window_stride_samples) + int(window_size_samples)]
        # print(audio_data.shape)
        # print(audio_data[:10])

        # cal feature
        feature = Feature(sample_rate, window_size_samples/sample_rate, int(args.CHW_params.split(",")[2]))
        feature.get_mel_int_feature(audio_data, len(audio_data))
        feature_data = feature.copy_mfsc_feature_int_to()
        # print(np.expand_dims(feature_data, axis=0).shape)

        # net forward
        detected_number = 0
        kws_stride_times = int((feature_data.shape[0] - kws_feature_time) * 1.0 / kws_stride_feature_time) + 1
        for kws_stride_time in range(kws_stride_times):
            feature_data_kws = feature_data[kws_stride_time * int(kws_stride_feature_time): kws_stride_time * int(kws_stride_feature_time) + int(kws_feature_time),:]
            feature_data_kws = feature_data_kws.astype(np.float32)
            # print(np.expand_dims(feature_data_kws, axis=0).shape)
            # print(feature_data_kws)

            net.blobs[args.net_input_name].data[...] = np.expand_dims(feature_data_kws, axis=0)
            net_output = net.forward()[args.net_output_name]
            net_output = np.squeeze(net_output)
            net_output = net_output.T
            # print(net_output.shape)
            print(net_output)

            if net_output[-1] > kws_detection_threshold:
                detected_number += 1

        if detected_number >= kws_stride_times * kws_detection_number_threshold:
            print("[Information:] Time: {}, Find Weakup word".format(times * int(window_stride_samples) / sample_rate))
            wave_loader.save_data(audio_data, os.path.join(args.output_folder, "kws_start_time_{}.wav".format(times * int(window_stride_samples) / sample_rate)))

def demo(args):
    global net, wave_loader
    global kws_feature_time, kws_stride_feature_time
    global sample_rate, kws_detection_threshold, kws_detection_number_threshold

    # init 
    kws_feature_time = int(args.CHW_params.split(",")[1])
    kws_stride_feature_time = 10
    kws_detection_threshold = 0.8
    kws_detection_number_threshold = 0.5
    sample_rate = 16000
    window_size_ms = 3000
    window_stride_ms = 1000
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    # mkdir 
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        
    # model init
    net = model_init(args.prototxt_path, args.model_path, args.net_input_name, args.CHW_params.split(","), args.gpu)

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

        decode_sliding_window(args, wave_data, window_size_samples, window_stride_samples)

if __name__ == "__main__":
    default_audio_folder = "/home/huanyuan/share/audio_data/weakup_xiaorui/"
    default_output_folder = "/home/huanyuan/share/audio_data/weakup_xiaorui/result_caffe"
    default_model_path = "/mnt/huanyuan/model/audio_model/caffe_model/kws_xiaorui_res15/res15_03162011.caffemodel"
    default_prototxt_path = "/mnt/huanyuan/model/audio_model/caffe_model/kws_xiaorui_res15/res15_03162011.prototxt"
    default_net_input_name = "blob1"
    default_net_output_name = "Softmax"
    default_CHW_params = "1,196,64"
    default_gpu = True

    parser = argparse.ArgumentParser(description='Streamax ASR Demo Engine')
    parser.add_argument('-i', '--audio_folder', type=str, default=default_audio_folder)
    parser.add_argument('-o', '--output_folder', type=str, default=default_output_folder)
    parser.add_argument('-m', '--model_path', type=str, default=default_model_path)
    parser.add_argument('-p', '--prototxt_path', type=str, default=default_prototxt_path)
    parser.add_argument('--net_input_name', type=str, default=default_net_input_name)
    parser.add_argument('--net_output_name', type=str, default=default_net_output_name)
    parser.add_argument('--CHW_params', type=str, default=default_CHW_params)
    parser.add_argument('-g', '--gpu', action='store_true', default=default_gpu)
    args = parser.parse_args()

    demo(args)