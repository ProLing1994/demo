import argparse
import cv2
import numpy as np
import os
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


def demo(args):
    # init 
    n_fft = 512
    feature_freq = 48
    time_seg_ms = 32
    time_step_ms = 10

    window_size_ms = 3000
    window_stride_ms = 2000
    sample_rate = 16000
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    # result id init
    result_id_list = []
    with open(args.result_dict, "r") as f :
        lines = f.readlines()       # 逐行读取文本文件（这种方法不好）
        for line in lines:          # 对每一行数据进行操作
            result_id_list.append(line.strip())

    # model init
    net = model_init(args.prototxt_path, args.model_path, args.CHW_params.split(","), args.gpu)

    wave_list = os.listdir(args.audio_folder)
    for idx in range(len(wave_list)):

        wave_path = os.path.join(args.audio_folder, wave_list[idx])
        print("[Information:] Audio path: ", wave_path)
        
        # load wave
        wave_loader = WaveLoader()
        wave_loader.load_data(wave_path)
        wave_data = wave_loader.to_numpy()

        # forward
        windows_times = int((len(wave_data) - window_size_samples) * 1.0 / window_stride_samples)
        for times in range(windows_times):
            print("\n[Information:] Wave start time: ", times * window_stride_samples)

            audio_data = wave_data[times * int(window_stride_samples): times * int(window_stride_samples) + int(window_size_samples)]
            feature_data = Feature.GetMelIntFeature(audio_data, len(audio_data), n_fft, sample_rate, time_seg_ms, time_step_ms, feature_freq)
            # print(feature_data)

            feature_data = feature_data.astype(np.float32)
            net.blobs['data'].data[...] = np.expand_dims(feature_data, axis=0)
            net_output = net.forward()['conv7']
            net_output = np.squeeze(net_output)
            net_output = net_output.T
            print(net_output.shape)
            # print(net_output[0])

            decode = Decode()
            decode.ctc_decoder(net_output)
            result_id = decode.result_id_to_numpy()
            print(result_id)
            
            result_string = ""
            for idx in range(len(result_id)):
                result_string += result_id_list[result_id[idx]]
                result_string += " "
            print(result_string)
            # cv2.imshow("feature_data",feature_data)
            # cv2.waitKey() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Streamax KWS Training Engine')
    parser.add_argument('-i', '--audio_folder', type=str, default="/home/huanyuan/share/audio_data/")
    parser.add_argument('-m', '--model_path', type=str, default="/mnt/huanyuan/model/kws_model/asr/mandarin_asr_nofc_16K.caffemodel")
    parser.add_argument('-p', '--prototxt_path', type=str, default="/mnt/huanyuan/model/kws_model/asr/mandarin_asr_nofc_16K.prototxt")
    parser.add_argument('--result_dict', type=str, default="/home/huanyuan/share/KWS_model/configFiles/dict_without_tone.txt")
    parser.add_argument('--CHW_params', type=str, default="1,296,48")
    parser.add_argument('-g', '--gpu', action='store_true', default=False)
    args = parser.parse_args()

    demo(args)