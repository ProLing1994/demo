import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys

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


def demo(args):
    # model init
    net = model_init(args.prototxt_path, args.model_path, args.net_input_name, args.CHW_params.split(","), args.gpu)

    # load image 
    feature_data = cv2.imread(args.filepath, 0) 
    # feature_data = np.ones(feature_data.shape, np.uint8)
    print(feature_data.shape)
    
    net.blobs[args.net_input_name].data[...] = np.expand_dims(feature_data, axis=0)

    net_output = net.forward()[args.net_output_name]
    net_output = np.squeeze(net_output)
    net_output = net_output.T
    print(net_output.shape)
    for idx in range(37):
        print("i: {}, sum:{}, data[0]:{}".format(idx, net_output[idx].sum(), net_output[idx][0]))


if __name__ == "__main__":
    # # chinese:
    # default_filepath = "/home/huanyuan/share/audio_data/RM_Carchat_Mandarin_P01020_0.png"
    # default_model_path = "/home/huanyuan/share/novt/KWS_model/asr_mandarin_16K.caffemodel"
    # default_prototxt_path = "/home/huanyuan/share/novt/KWS_model/asr_mandarin_16K.prototxt"
    # default_net_input_name = "blob1"
    # default_net_output_name = "prob"
    # default_CHW_params = "1,296,56"
    # default_gpu = False
    
    # test:
    default_filepath = "/home/huanyuan/share/audio_data/RM_Carchat_Mandarin_P01020_0.png"
    default_model_path = "/home/huanyuan/share/novt/KWS_model/asr_mandarin_16K.caffemodel"
    # default_prototxt_path = "/home/huanyuan/share/novt/KWS_model/asr_mandarin_16K_test.prototxt"
    default_prototxt_path = "/home/huanyuan/share/novt/KWS_model/asr_mandarin_16K_update.prototxt"
    default_net_input_name = "blob1"
    default_net_output_name = "prob"
    default_CHW_params = "1,296,56"
    default_gpu = False

    parser = argparse.ArgumentParser(description='Streamax ASR Demo Engine')
    parser.add_argument('-i', '--filepath', type=str, default=default_filepath)
    parser.add_argument('-m', '--model_path', type=str, default=default_model_path)
    parser.add_argument('-p', '--prototxt_path', type=str, default=default_prototxt_path)
    parser.add_argument('--net_input_name', type=str, default=default_net_input_name)
    parser.add_argument('--net_output_name', type=str, default=default_net_output_name)
    parser.add_argument('--CHW_params', type=str, default=default_CHW_params)
    parser.add_argument('-g', '--gpu', action='store_true', default=default_gpu)
    args = parser.parse_args()

    demo(args)