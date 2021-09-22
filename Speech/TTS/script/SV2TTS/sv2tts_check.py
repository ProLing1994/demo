import argparse
import numpy as np
import os
import sys
import shutil
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from KWS.demo.network.ASR_english_phoneme import ASR_English_Net

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_feature_pyimpl import Feature
import ASR.impl.asr_decode_pyimpl as Decode_Python
import ASR.impl.asr_data_loader_pyimpl as WaveLoader_Python

caffe_root = "/home/huanyuan/code/caffe_ssd/"
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')
import caffe

import torch

def pytorch_model_init(args): 
    # init model 
    net = ASR_English_Net(args.asr_num_classes)

    if args.use_gpu:
        net = net.cuda()

    # load state
    checkpoint=torch.load(args.asr_chk_file)
    net.load_state_dict(checkpoint['state_dict'], strict=True)

    net.eval()
    return net

def pytorch_model_forward(net, feature_data, use_gpu=False):
    data_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(feature_data, axis=0), axis=0))
    data_tensor = data_tensor.float()

    if use_gpu:
        data_tensor = data_tensor.cuda()
    net_output = net(data_tensor).cpu().data.numpy()
    return net_output

def caffe_model_init(prototxt, model, input_name, CHW_params, use_gpu=False):
    if use_gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
        print("[Information:] GPU mode")
    else:
        caffe.set_mode_cpu()
        print("[Information:] CPU mode")
    net = caffe.Net(prototxt, model, caffe.TEST)
    # net.blobs[input_name].reshape(1, int(CHW_params[0]), int(CHW_params[1]), int(CHW_params[2])) 
    return net

def caffe_model_forward(net, feature_data, input_name, output_name, bool_transpose=False):
    if bool_transpose:
        feature_data = feature_data.T

    net.blobs[input_name].reshape(1, 1, feature_data.shape[0], feature_data.shape[1])
    net.blobs[input_name].data[...] = np.expand_dims(np.expand_dims(feature_data, axis=0), axis=0)
    net_output = net.forward()[output_name]
    return net_output

def ASR_offine(args):
    # mkdir 
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if not os.path.exists(args.output_remove_folder):
        os.makedirs(args.output_remove_folder)

    # init model 
    if args.bool_caffe:
        net = caffe_model_init(args.asr_prototxt_path, args.asr_model_path, args.asr_net_input_name, args.asr_chw_params.split(","), args.use_gpu)
    else:
        net = pytorch_model_init(args)

    # init dict 
    decode_python = Decode_Python.Decode()
    decode_python.init_symbol_list(args.asr_dict_path)

    data_list = os.listdir(args.input_folder)
    data_list.sort()
    for data_idx in tqdm(range(len(data_list))):
        if not data_list[data_idx].endswith(".wav"):
            continue

        wave_path = os.path.join(args.input_folder, data_list[data_idx])
        
        # load wave
        wave_loader = WaveLoader_Python.WaveLoader_Librosa(args.sample_rate)
        wave_loader.load_data(wave_path)
        wave_data = wave_loader.to_numpy()

        # feature
        feature = Feature(args.sample_rate, len(wave_data)/args.sample_rate, int(args.feature_freq), int(args.nfilt))
        feature.get_mel_int_feature(wave_data, len(wave_data))
        feature_data = feature.copy_mfsc_feature_int_to()

        # forward
        if args.bool_caffe:
            net_output = caffe_model_forward(net, feature_data, args.asr_net_input_name, args.asr_net_output_name)
            net_output = np.squeeze(net_output)
            net_output = net_output.T
        else:
            net_output = pytorch_model_forward(net, feature_data, args.use_gpu)
            net_output = np.squeeze(net_output)
        
        # decode
        decode_python.ctc_decoder(net_output)
        # decode_python.show_result_id()
        # decode_python.show_symbol()
        decode_python.show_symbol_english()

        output_string = decode_python.output_symbol_english()
        find_bool = True in [args.check_list[idx] in output_string for idx in range(len(args.check_list))]
        # print(find_bool)

        if find_bool:
            output_path = os.path.join(args.output_folder, data_list[data_idx])
            shutil.copy(wave_path, output_path)
        else:
            output_path = os.path.join(args.output_remove_folder, data_list[data_idx])
            shutil.copy(wave_path, output_path)


def main():
    parser = argparse.ArgumentParser(description='Streamax KWS Data Split Engine')
    args = parser.parse_args()

    # phoneme
    # args.bool_caffe = False
    # args.asr_chk_file = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_phoneme_16k_06082021/asr_english_phoneme_16k_64_0608.pth"
    # args.asr_dict_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_phoneme_16k_06032021/asr_english_phoneme_dict.txt"
    # args.asr_num_classes = 136

    # bpe
    args.bool_caffe = True
    args.asr_model_path= "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_16k_0202/english_0202_better.caffemodel"
    args.asr_prototxt_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_16k_0202/english_0202_mark.prototxt"
    args.asr_net_input_name = "data"
    args.asr_net_output_name = "conv39"
    args.asr_chw_params = "1,296,64"
    args.asr_dict_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_16k_0202/english_bpe.txt"
    args.sample_rate = 16000
    args.feature_freq = 64
    args.nfilt = 64
    args.use_gpu = True

    # args.input_folder = "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/RM_Meiguo_BwcKeyword/danbing_16k/wav/"
    # args.output_folder = "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/RM_Meiguo_BwcKeyword/danbing_16k/wav_check/"
    # args.output_remove_folder = "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/RM_Meiguo_BwcKeyword/danbing_16k/wav_check_remove/"
    args.input_folder = "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/LibriSpeech/train-other-500/activatebwc/"
    args.output_folder = "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/LibriSpeech/train-other-500_check/activatebwc/"
    args.output_remove_folder = "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/LibriSpeech/train-other-500_check_remove/activatebwc/"
    args.check_list = ['activate', ' be ', 'double', 'you', 'see', 'say']

    ASR_offine(args)


if __name__ == "__main__":
    """
    # 利用 ASR 模型，对 SV2TTS 系统生成的数据，进行检查，挑选出合理数据
    """
    main()

