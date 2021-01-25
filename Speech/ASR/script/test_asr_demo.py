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

caffe_root = '/home/huanyuan/code/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe


def model_init(prototxt, model, CHW_params, use_gpu=False):
    if use_gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
        print("[Information:] GPU mode")
    else:
        caffe.set_mode_cpu()
        print("[Information:] CPU mode")
    net = caffe.Net(prototxt, model, caffe.TEST)
    net.blobs['data'].reshape(1, int(CHW_params[0]), int(CHW_params[1]), int(CHW_params[2])) 
    return net


def demo_total_window(args):
    # init 
    sample_rate = 16000

    # result id init
    result_id_list = []
    with open(args.result_dict, "r") as f :
        lines = f.readlines()       
        for line in lines:          
            result_id_list.append(line.strip())

    # model init
    net = model_init(args.prototxt_path, args.model_path, args.CHW_params.split(","), args.gpu)

    result_list = []
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

        # forward
        result_string = ""
        
        # total window
        audio_data = wave_data

        # cal feature
        feature = Feature(len(audio_data))
        feature.get_mel_int_feature(audio_data, len(audio_data))
        feature_data = feature.copy_mfsc_feature_int_to()

        # net forward
        feature_data = feature_data.astype(np.float32)
        # reshape
        net.blobs['data'].reshape(1, 1, feature_data.shape[0], feature_data.shape[1])
        net.blobs['data'].data[...] = np.expand_dims(feature_data, axis=0)

        # print(np.expand_dims(feature_data, axis=0).shape)
        # print(feature_data)

        net_output = net.forward()['conv7']
        # print(net_output.shape)
        net_output = np.squeeze(net_output)
        net_output = net_output.T
        # print(net_output.shape)
        # print(net_output[0])

        # decode
        decode = Decode()
        decode.ctc_decoder(net_output)
        result_id = decode.result_id_to_numpy()
        # print(result_id)
        
        for result_idx in range(len(result_id)):
            result_string += result_id_list[result_id[result_idx]]
            result_string += " "

        print(result_string)
        result_dict = {}
        result_dict['data'] = wave_list[idx]
        result_dict['caffe_result'] = (result_string)
        result_list.append(result_dict)
    result_pd = pd.DataFrame(result_list, columns=['data', 'caffe_result'])

    output_folder = args.output_folder

    # mkdir 
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    result_pd.to_csv(os.path.join(output_folder, 'result_total_window.csv'), index=False)


def demo_sliding_window(args):
    # init 
    window_size_ms = 3000
    window_stride_ms = 2000
    sample_rate = 16000
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    # result id init
    result_id_list = []
    with open(args.result_dict, "r") as f :
        lines = f.readlines()       
        for line in lines:          
            result_id_list.append(line.strip())

    # model init
    net = model_init(args.prototxt_path, args.model_path, args.CHW_params.split(","), args.gpu)

    result_list = []
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

        # forward
        result_string = ""
        
        # sliding window
        windows_times = int((len(wave_data) - window_size_samples) * 1.0 / window_stride_samples) + 1
        for times in range(windows_times):
            print("\n[Information:] Wave start time: ", times * window_stride_samples)

            audio_data = wave_data[times * int(window_stride_samples): times * int(window_stride_samples) + int(window_size_samples)]

            # cal feature
            feature = Feature()
            feature.get_mel_int_feature(audio_data, len(audio_data))
            feature_data = feature.copy_mfsc_feature_int_to()

            # net forward
            feature_data = feature_data.astype(np.float32)
            net.blobs['data'].data[...] = np.expand_dims(feature_data, axis=0)

            # print(np.expand_dims(feature_data, axis=0).shape)
            # print(feature_data)

            net_output = net.forward()['conv7']
            # print(net_output.shape)
            net_output = np.squeeze(net_output)
            net_output = net_output.T
            # print(net_output.shape)
            # print(net_output[0])

            # decode
            decode = Decode()
            decode.ctc_decoder(net_output)
            result_id = decode.result_id_to_numpy()
            # print(result_id)
            
            for result_idx in range(len(result_id)):
                result_string += result_id_list[result_id[result_idx]]
                result_string += " "

        print(result_string)
        result_dict = {}
        result_dict['data'] = wave_list[idx]
        result_dict['caffe_result'] = (result_string)
        result_list.append(result_dict)
    result_pd = pd.DataFrame(result_list, columns=['data', 'caffe_result'])

    output_folder = args.output_folder

    # mkdir 
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    result_pd.to_csv(os.path.join(output_folder, 'result_sliding_window.csv'), index=False)


if __name__ == "__main__":
    bool_total_window = False
    parser = argparse.ArgumentParser(description='Streamax KWS Training Engine')
    parser.add_argument('-i', '--audio_folder', type=str, default="/home/huanyuan/share/audio_data/第三批数据/安静场景/")
    parser.add_argument('-o', '--output_folder', type=str, default="/home/huanyuan/share/audio_data/第三批数据/安静场景/result_caffe")
    parser.add_argument('-m', '--model_path', type=str, default="/mnt/huanyuan/model/kws_model/asr/mandarin_asr_nofc_16K.caffemodel")
    parser.add_argument('-p', '--prototxt_path', type=str, default="/home/huanyuan/share/KWS_model/mandarin_asr_nofc_16K.prototxt")
    parser.add_argument('--result_dict', type=str, default="/home/huanyuan/share/KWS_model/configFiles/dict_without_tone.txt")
    parser.add_argument('--CHW_params', type=str, default="1,296,48")
    parser.add_argument('-g', '--gpu', action='store_true', default=False)
    args = parser.parse_args()

    if bool_total_window:
        demo_total_window(args)
    else:
        demo_sliding_window(args)
