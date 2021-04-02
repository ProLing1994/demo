import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/ASR')
from impl.asr_data_loader_cimpl import WaveLoader
# from impl.asr_feature_cimpl import Feature
# from impl.asr_data_loader_pyimpl import WaveLoader
from impl.asr_feature_pyimpl import Feature
from impl.asr_decode_cimpl import Decode

caffe_root = '/home/huanyuan/code/caffe/'
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


def output_symbol_chinese(result_id):
    global result_symbol_list
    
    output_symbol = ""
    for idx in range(len(result_id)):
        output_symbol += result_symbol_list[result_id[idx]]
        output_symbol += " "
    return output_symbol


def output_symbol_english(result_id):
    global result_symbol_list

    output_symbol = ""
    for idx in range(len(result_id)):
        symbol = result_symbol_list[result_id[idx]] 

        if symbol[0] == '_':
            if idx != 0:
                output_symbol += " "
            output_symbol += symbol[1:]
        else:
            output_symbol += symbol
    if len(result_id):
        output_symbol += " "
    return output_symbol


def decode_full_window(args, wave_data):
    global language
    global net

    # forward
    result_string = ""
    
    # data
    audio_data = wave_data

    # cal feature
    # feature = Feature(len(audio_data))
    feature = Feature(sample_rate, len(audio_data)/sample_rate, int(args.CHW_params.split(",")[2]))
    feature.get_mel_int_feature(audio_data, len(audio_data))
    feature_data = feature.copy_mfsc_feature_int_to()
    # print(np.expand_dims(feature_data, axis=0).shape)
    # print(feature_data)

    # net forward
    feature_data = feature_data.astype(np.float32)
    net.blobs[args.net_input_name].reshape(1, 1, feature_data.shape[0], feature_data.shape[1])
    net.blobs[args.net_input_name].data[...] = np.expand_dims(feature_data, axis=0)

    net_output = net.forward()[args.net_output_name]
    net_output = np.squeeze(net_output)
    net_output = net_output.T
    # print(net_output.shape)
    # print(net_output[0])

    # decode
    decode = Decode()
    decode.ctc_decoder(net_output)
    result_id = decode.result_id_to_numpy()
    # print(result_id)
    
    # [0: chinese
    #  1: english]
    if language == 0:
        result_string = output_symbol_chinese(result_id)
    elif language == 1:
        result_string = output_symbol_english(result_id)
    else:
        raise Exception("[ERROR: ] Unknow language")
    return result_string


def decode_sliding_window(args, wave_data, window_size_samples, window_stride_samples):
    global language
    global net

    # forward
    result_string = ""

    # sliding window
    windows_times = int((len(wave_data) - window_size_samples) * 1.0 / window_stride_samples) + 1
    for times in range(windows_times):
        print("[Information:] Wave start time: ", times * window_stride_samples)

        audio_data = wave_data[times * int(window_stride_samples): times * int(window_stride_samples) + int(window_size_samples)]
        print(audio_data.shape)
        print(audio_data[:10])

        # cal feature
        # feature = Feature()
        feature = Feature(sample_rate, window_size_samples/sample_rate, int(args.CHW_params.split(",")[2]))
        feature.get_mel_int_feature(audio_data, len(audio_data))
        feature_data = feature.copy_mfsc_feature_int_to()
        print(np.expand_dims(feature_data, axis=0).shape)
        # print(feature_data)
        # print(feature_data.sum())

        # net forward
        feature_data = feature_data.astype(np.float32)
        net.blobs[args.net_input_name].data[...] = np.expand_dims(feature_data, axis=0)

        net_output = net.forward()[args.net_output_name]
        net_output = np.squeeze(net_output)
        net_output = net_output.T
        print(net_output.shape)
        # print(net_output[0])

        # decode
        decode = Decode()
        decode.ctc_decoder(net_output)
        result_id = decode.result_id_to_numpy()
        print(result_id)
        
        # [0: chinese
        #  1: english]
        if language == 0:
            window_result_string = output_symbol_chinese(result_id)
        elif language == 1:
            window_result_string = output_symbol_english(result_id)
            # window_result_string = output_symbol_chinese(result_id)
        else:
            raise Exception("[ERROR: ] Unknow language")
        
        print(window_result_string)
        result_string += window_result_string

    return result_string


def demo(args):
    global mode
    global net
    global result_symbol_list

    # init 
    sample_rate = 16000
    window_size_ms = 3000
    window_stride_ms = 2000
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    # result id init
    result_symbol_list = []
    with open(args.bpe, "r") as f :
        lines = f.readlines()       
        for line in lines:          
            result_symbol_list.append(line.strip())

    # model init
    net = model_init(args.prototxt_path, args.model_path, args.net_input_name, args.CHW_params.split(","), args.gpu)

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

        # [0: Sliding window decoding
        #  1: Full window decoding]
        if mode == 0:
            result_string = decode_sliding_window(args, wave_data, window_size_samples, window_stride_samples)
        elif mode == 1:
            result_string = decode_full_window(args, wave_data)
        else:
            raise Exception("[ERROR: ] Unknow mode")

        print("[Information:] outKeyword: ", result_string)
        result_dict = {}
        result_dict['data'] = wave_list[idx]
        result_dict['caffe_result'] = (result_string)
        result_list.append(result_dict)
    result_pd = pd.DataFrame(result_list, columns=['data', 'caffe_result'])

    output_folder = args.output_folder

    # mkdir 
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if mode == 0:
        result_pd.to_csv(os.path.join(output_folder, 'result_sliding_window.csv'), index=False)
    elif mode == 1:
        result_pd.to_csv(os.path.join(output_folder, 'result_full_window.csv'), index=False)
    else:
        raise Exception("[ERROR: ] Unknow mode")


if __name__ == "__main__":
    # mode: 
    # [0: Sliding window decoding
    #  1: Full window decoding]
    global mode
    mode = 0

   # language: 
    # [0: chinese
    #  1: english]
    global language
    language = 0

    # chinese (amba):
    # default_audio_folder = "/home/huanyuan/share/audio_data/第三批数据/安静场景/"
    # default_output_folder = "/home/huanyuan/share/audio_data/第三批数据/安静场景/result_caffe"
    default_audio_folder = "/home/huanyuan/share/audio_data/"
    default_output_folder = "/home/huanyuan/share/audio_data/result_caffe"
    default_model_path = "/home/huanyuan/share/amba/KWS_model/mandarin_asr_nofc_16K.caffemodel"
    default_prototxt_path = "/home/huanyuan/share/amba/KWS_model/mandarin_asr_nofc_16K.prototxt"
    default_bpe = "/home/huanyuan/share/amba/KWS_model/configFiles/dict_without_tone.txt"
    default_net_input_name = "data"
    default_net_output_name = "conv7"
    default_CHW_params = "1,296,48"
    default_gpu = False

    # # chinese (novt):
    # # default_audio_folder = "/home/huanyuan/share/audio_data/"
    # # default_output_folder = "/home/huanyuan/share/audio_data/result_caffe"
    # default_audio_folder = "/home/huanyuan/share/audio_data/第三批数据/闹市场景/"
    # default_output_folder = "/home/huanyuan/share/audio_data/第三批数据/闹市场景/result_caffe"
    # default_model_path = "/home/huanyuan/share/novt/KWS_model/asr_mandarin_16K_20210302.caffemodel"
    # default_prototxt_path = "/home/huanyuan/share/novt/KWS_model/asr_mandarin_16K_20210302.prototxt"
    # # default_model_path = "/home/huanyuan/share/novt/KWS_model/asr_mandarin_16K_0304.caffemodel"
    # # default_prototxt_path = "/home/huanyuan/share/novt/KWS_model/asr_mandarin_16K_0304.prototxt"
    # default_bpe = "/home/huanyuan/share/novt/KWS_model/configFiles/dict_without_tone.txt"
    # default_net_input_name = "blob1"
    # default_net_output_name = "conv_blob24"
    # default_CHW_params = "1,296,56"
    # default_gpu = False

    # # english:
    # default_audio_folder = "/home/huanyuan/share/audio_data/english_wav/"
    # default_output_folder = "/home/huanyuan/share/audio_data/english_wav/result_caffe"
    # default_model_path = "/mnt/huanyuan/model/kws_model/asr_english/english_0202_better.caffemodel"
    # default_prototxt_path = "/mnt/huanyuan/model/kws_model/asr_english/english_0202_mark.prototxt"
    # default_bpe = "/home/huanyuan/share/KWS_model/configFiles/english_bpe.txt"
    # default_net_input_name = "data"
    # default_net_output_name = "conv39"
    # default_CHW_params = "1,296,64"
    # default_gpu = False

    parser = argparse.ArgumentParser(description='Streamax ASR Demo Engine')
    parser.add_argument('-i', '--audio_folder', type=str, default=default_audio_folder)
    parser.add_argument('-o', '--output_folder', type=str, default=default_output_folder)
    parser.add_argument('-m', '--model_path', type=str, default=default_model_path)
    parser.add_argument('-p', '--prototxt_path', type=str, default=default_prototxt_path)
    parser.add_argument('--bpe', type=str, default=default_bpe)
    parser.add_argument('--net_input_name', type=str, default=default_net_input_name)
    parser.add_argument('--net_output_name', type=str, default=default_net_output_name)
    parser.add_argument('--CHW_params', type=str, default=default_CHW_params)
    parser.add_argument('-g', '--gpu', action='store_true', default=default_gpu)
    args = parser.parse_args()

    demo(args)