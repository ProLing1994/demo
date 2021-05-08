import argparse
from datetime import datetime
import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_data_loader_pyimpl import WaveLoader
# from ASR.impl.asr_data_loader_cimpl import WaveLoader
from ASR.impl.asr_feature_pyimpl import Feature
# from ASR.impl.asr_feature_cimpl import Feature
import ASR.impl.asr_decode_cimpl as Decode_C
import ASR.impl.asr_decode_pyimpl as Decode_Python
import KWS.demo.RMAI_KWS_ASR_options as cfg

caffe_root = "/home/huanyuan/code/caffe_ssd/"
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')
import caffe

# options 
cfg = cfg.cfg

# params
# container
audio_data_container_np = np.array([])
feature_data_container_np = np.array([])
kws_container_np = np.array([])         # kws 结构容器中，用于滑窗输出结果
output_wave_list = []

bool_weakup = False
counter_weakup = 0
counter_asr = 0
output_kws_id = 1

# argparse
# kws
# activatebwc
# default_kws_model_path = "/mnt/huanyuan/model/audio_model/caffe_model/kws_activatebwc_res15/res15_1_5_03302021.caffemodel"
# default_kws_prototxt_path = "/mnt/huanyuan/model/audio_model/caffe_model/kws_activatebwc_res15/res15_1_5_03302021.prototxt"
# default_kws_label = "activatebwc"
# default_kws_net_input_name = "blob1"
# default_kws_net_output_name = "Softmax"
# default_kws_chw_params = "1,196,64"
# default_kws_transpose = False
default_kws_model_path = "/mnt/huanyuan/model/audio_model/amba_model/kws_activatebwc_tc_resnet14/tc_resnet14_amba_2_4_04012021.caffemodel"
default_kws_prototxt_path = "/mnt/huanyuan/model/audio_model/amba_model/kws_activatebwc_tc_resnet14/tc_resnet14_amba_2_4_04012021.prototxt"
default_kws_label = "activatebwc"
default_kws_net_input_name = "data"
default_kws_net_output_name = "Softmax"
default_kws_chw_params = "1,64,196"
default_kws_transpose = True

# # xiaoan8k
# # default_kws_model_path = "/mnt/huanyuan/model/audio_model/caffe_model/kws_xiaoan8k_res15/res15_1_1_04062021.caffemodel"
# # default_kws_prototxt_path = "/mnt/huanyuan/model/audio_model/caffe_model/kws_xiaoan8k_res15/res15_1_1_04062021.prototxt"
# # default_kws_label = "xiaoan8k"
# # default_kws_net_input_name = "blob1"
# # default_kws_net_output_name = "Softmax"
# # default_kws_chw_params = "1,146,48"
# # default_kws_transpose = False
# default_kws_model_path = "/mnt/huanyuan/model/audio_model/caffe_model/kws_xiaoan8k_tc_resnet14/tc_resnet14_2_2_04162021.caffemodel"
# default_kws_prototxt_path = "/mnt/huanyuan/model/audio_model/caffe_model/kws_xiaoan8k_tc_resnet14/tc_resnet14_2_2_04162021.prototxt"
# default_kws_label = "xiaoan8k"
# default_kws_net_input_name = "data"
# default_kws_net_output_name = "Softmax"
# default_kws_chw_params = "1,48,146"
# default_kws_transpose = True

# asr
default_asr_model_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/english_0202_better.caffemodel"
default_asr_prototxt_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/english_0202_mark.prototxt"
default_asr_net_input_name = "data"
default_asr_net_output_name = "conv39"
default_asr_chw_params = "1,296,64"
default_asr_bpe = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/english_bpe.txt"

# test_mode
# 0: input_wav
# 1: input_folder
default_test_mode = 1

# input_Wav
# activatbwc16k
# default_input_wav = "/home/huanyuan/share/audio_data/english_wav/1-0127-asr_16k.wav"
# default_input_wav = "/mnt/huanyuan/model/test_straming_wav/activatebwc_1_5_03312021_validation_180.wav"
# default_input_wav = "/mnt/huanyuan/data/speech/Recording_sample/Jabra_510/Jabra_510_test-kws-asr_0001.wav"
# default_input_wav = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/test_dataset/海外同事录制_0425/安静场景/场景一/RM_KWS_ACTIVATEBWC_ovweseas_ori_S010M0D00T2.wav"

# xiaoan8k
# default_input_wav = "/mnt/huanyuan/model/test_straming_wav/xiaoan8k_1_1_04082021_validation_60.wav"
# default_input_wav = "/mnt/huanyuan/model/test_straming_wav/xiaoan8k_1_3_04152021_validation.wav"
# default_input_wav = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/自测_0412/安静场景/xiaoan_麦克风_场景二_31.wav"
# default_input_wav = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/实车场景/处理音频/吴毅然_adpro.wav"
default_input_wav = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/其他录音/adpro/B9P01D_20201113_231117_C02_Main_02_02.wav"

# input_folder
# default_input_folder = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/test_dataset/海外同事录制_0425/路边场景/场景二/"
# default_input_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/实车场景/处理音频/"
# default_input_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/其他录音/adpro/"
# default_input_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/其他录音/adpro-1/"
default_input_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/其他录音/adpro-2/"
# default_input_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/其他录音/test/"

default_output_folder = "/mnt/huanyuan/data/speech/Recording_sample/demo_kws_asr_online_api/{}".format('-'.join('-'.join(str(datetime.now()).split('.')[0].split(' ')).split(':')))
default_gpu = True

parser = argparse.ArgumentParser(description='Streamax KWS ASR offine Engine')
parser.add_argument('--test_mode', type=int, default=default_test_mode)
parser.add_argument('--input_wav', type=str, default=default_input_wav)
parser.add_argument('--input_folder', type=str, default=default_input_folder)
parser.add_argument('--kws_model_path', type=str, default=default_kws_model_path)
parser.add_argument('--kws_prototxt_path', type=str, default=default_kws_prototxt_path)
parser.add_argument('--kws_label', type=str, default=default_kws_label)
parser.add_argument('--kws_net_input_name', type=str, default=default_kws_net_input_name)
parser.add_argument('--kws_net_output_name', type=str, default=default_kws_net_output_name)
parser.add_argument('--kws_chw_params', type=str, default=default_kws_chw_params)
parser.add_argument('--kws_transpose', action='store_true', default=default_kws_transpose)
parser.add_argument('--asr_model_path', type=str, default=default_asr_model_path)
parser.add_argument('--asr_prototxt_path', type=str, default=default_asr_prototxt_path)
parser.add_argument('--asr_net_input_name', type=str, default=default_asr_net_input_name)
parser.add_argument('--asr_net_output_name', type=str, default=default_asr_net_output_name)
parser.add_argument('--asr_chw_params', type=str, default=default_asr_chw_params)
parser.add_argument('--asr_bpe', type=str, default=default_asr_bpe)
parser.add_argument('--output_folder', type=str, default=default_output_folder)
parser.add_argument('--gpu', action='store_true', default=default_gpu)
args = parser.parse_args()


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


def kws_asr_init():
    global kws_net, asr_net
    global decode_python

    # init model
    kws_net = model_init(args.kws_prototxt_path, args.kws_model_path, args.kws_net_input_name, args.kws_chw_params.split(","), args.gpu)
    asr_net = model_init(args.asr_prototxt_path, args.asr_model_path, args.asr_net_input_name, args.asr_chw_params.split(","), args.gpu)

    # init bpe
    decode_python = Decode_Python.Decode()
    decode_python.init_ast_symbol_list(args.asr_bpe)

    # mkdir
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    

def run_kws():
    # 采用滑窗的方式判断是否触发 kws
    # 否则直接在 1s 内直接查找 kws，会漏掉起始和结尾点，造成漏唤醒
    global kws_container_np

    # init
    kws_score_list = []

    if not cfg.general.bool_do_kws_weakup:
        return False, kws_score_list

    # 滑窗，模型前传
    kws_stride_times = int((feature_data_container_np.shape[0] - cfg.general.kws_feature_time) * 1.0 / cfg.general.kws_stride_feature_time) + 1
    for times in range(kws_stride_times - cfg.general.kws_overlap_detection_time):
        start_times = times + cfg.general.kws_overlap_detection_time
        feature_data_kws = feature_data_container_np[start_times * int(cfg.general.kws_stride_feature_time): start_times * int(cfg.general.kws_stride_feature_time) + int(cfg.general.kws_feature_time),:]
        feature_data_kws = feature_data_kws.astype(np.float32)
        # print(feature_data_kws)
        if args.kws_transpose:
            feature_data_kws = feature_data_kws.T
        kws_net.blobs[args.kws_net_input_name].data[...] = np.expand_dims(feature_data_kws, axis=0)

        net_output = kws_net.forward()[args.kws_net_output_name]
        net_output = np.squeeze(net_output)
        kws_score_list.append(net_output.copy())
        # print(times, feature_data_kws.shape, net_output)
        # print(net_output)

    # 如果有保留的 kws 结果，进行拼接
    kws_score_np = np.array(kws_score_list)
    if len(kws_container_np):
        kws_score_np = np.concatenate((kws_container_np, kws_score_np), axis=0)
    
    bool_find_kws = False
    for kws_idx in range(len(kws_score_np) + 1 - kws_stride_times):
        # 滑窗，获得后处理结果
        detected_number = 0 
        for kws_times in range(kws_stride_times):
            if kws_score_np[kws_idx + kws_times][-1] > cfg.general.kws_detection_threshold:
                detected_number += 1

        if detected_number >= kws_stride_times * cfg.general.kws_detection_number_threshold:
            bool_find_kws = True
    
    if bool_find_kws:
        kws_container_np = np.array([])
    else:
        # 存储一定时间的 kws 结果，用于后续滑窗获得结果
        kws_container_np = np.array(kws_score_list)

    return bool_find_kws, kws_score_list


def run_asr():
    if not cfg.general.bool_do_asr:
        return "cfg.general.bool_do_asr = False", "", ""

    feature_data_asr = feature_data_container_np.astype(np.float32)
    asr_net.blobs[args.asr_net_input_name].data[...] = np.expand_dims(feature_data_asr, axis=0)

    net_output = asr_net.forward()[args.asr_net_output_name]
    net_output = np.squeeze(net_output)
    net_output = net_output.T

    # decode
    decode_c = Decode_C.Decode()
    decode_c.ctc_decoder(net_output)
    result_id = decode_c.result_id_to_numpy()
    result_string = decode_python.output_symbol_english(result_id)
    matched_string = decode_python.match_kws_english(result_string.split(' '))
    control_command_string, not_control_command_string = decode_python.match_kws_english_control_command(matched_string)
    return result_string, control_command_string, not_control_command_string


def run_kws_asr(audio_data):
    global audio_data_container_np, feature_data_container_np
    global bool_weakup, counter_weakup, counter_asr
    global output_wave_list, output_kws_id
    global csv_original_scores, csv_found_words, subfolder_name
    global sliding_window_start_time_ms

    # 加载音频数据，用于打印输出
    if len(output_wave_list) < cfg.general.total_time_samples:
        output_wave_list.extend(audio_data)
    else:
        output_wave_list = output_wave_list[cfg.general.window_size_samples:]
        output_wave_list.extend(audio_data)

    # 如果有保留的音频数据，进行拼接
    if len(audio_data_container_np):
        audio_data = np.concatenate((audio_data_container_np, audio_data), axis=0)
    
    # 存储一定时间的音频，用于后续计算特征
    audio_data_container_np = audio_data[len(audio_data) - cfg.general.window_container_samples:]
    # print("[Information:] Audio length: ", len(audio_data), len(audio_data_container_np))

    # 计算特征
    feature = Feature(cfg.general.sample_rate, len(audio_data)/cfg.general.sample_rate, int(cfg.general.feature_freq), int(cfg.general.nfilt))
    feature.get_mel_int_feature(audio_data, len(audio_data))
    feature_data = feature.copy_mfsc_feature_int_to()
    # print("[Information:] Feature shape: ", feature_data.shape)

    # 更新特征
    if not feature_data_container_np.shape[0]:
        feature_data_container_np = feature_data
    elif feature_data_container_np.shape[0] < cfg.general.feature_container_time:
        feature_data_container_np = np.concatenate((feature_data_container_np[: -cfg.general.feature_remove_after_time], feature_data), axis=0)
    else:
        feature_data_container_np = np.concatenate((feature_data_container_np[cfg.general.feature_remove_before_time: -cfg.general.feature_remove_after_time], feature_data), axis=0)
    # print("[Information:] Feature container shape: ", feature_data_container_np.shape)

    # 如果语音特征未装满容器，不进行唤醒和关键词检测
    if feature_data_container_np.shape[0] < cfg.general.feature_container_time:
        return

    # 方案一：进行 kws 唤醒词检测，若检测到唤醒词，未来三秒进行 asr 检测
    # kws
    if not bool_weakup:
        bool_find_kws, kws_score_list = run_kws()

        if bool_find_kws:
            print("[Information:] Find Kws Weakup")
            bool_weakup = True

            # save audio
            if cfg.general.bool_output_wave and cfg.general.bool_output_csv:
                output_path = os.path.join(args.output_folder, subfolder_name, 'label_{}_starttime_{}.wav'.format(args.kws_label, int(sliding_window_start_time_ms)))
                wave_loader = WaveLoader(cfg.general.sample_rate)
                wave_loader.save_data(np.array(output_wave_list), output_path)

                csv_found_words.append({'label':args.kws_label, 'start_time':int(sliding_window_start_time_ms), 'end_time': int(sliding_window_start_time_ms + cfg.general.total_time_ms)})
                
            elif cfg.general.bool_output_wave:
                wave_loader = WaveLoader(cfg.general.sample_rate)
                date_time = '-'.join('-'.join(str(datetime.now()).split('.')[0].split(' ')).split(':'))
                wave_loader.save_data(np.array(output_wave_list), os.path.join(args.output_folder, "Weakup_{}_{:0>4d}.wav".format(date_time, output_kws_id)))
                output_kws_id += 1

        if cfg.general.bool_output_csv:
            for idx in range(len(kws_score_list) - 1):
                csv_original_scores.append({'start_time':sliding_window_start_time_ms + (idx + cfg.general.kws_overlap_detection_time) * cfg.general.kws_stride_feature_time * 10, 'score':",".join([str(kws_score_list[idx][idy]) for idy in range(kws_score_list[idx].shape[0])])})

    else:
        counter_weakup += 1
        if counter_weakup == cfg.general.kws_suppression_counter:
            counter_weakup = 0
            bool_weakup = False
            # 控制 asr 的间隔时间
            counter_asr -= 1

            # asr
            result_string, control_command_string, _ = run_asr()
            print("[Information:] kws asr outKeyword: ", result_string)
            # print("[Information:] kws asr outKeyword: ", control_command_string)
        
        if cfg.general.bool_output_csv:
            _, kws_score_list = run_kws()
            for idx in range(len(kws_score_list) - 1):
                csv_original_scores.append({'start_time':sliding_window_start_time_ms + (idx + cfg.general.kws_overlap_detection_time) * cfg.general.kws_stride_feature_time * 10, 'score':",".join([str(kws_score_list[idx][idy]) for idy in range(kws_score_list[idx].shape[0])])})

    # 方案二：进行 asr 检测，间隔一定时长
    # asr
    # 如果检测到唤醒词，则执行方案一
    if bool_weakup:
        counter_asr = 0
    else:
        counter_asr += 1

    if counter_asr == cfg.general.asr_suppression_counter:
        counter_asr = 0

        result_string, _, not_control_command_string = run_asr()
        print("[Information:] asr outKeyword: ", result_string)
        # print("[Information:] asr outKeyword: ", not_control_command_string)



def KWS_ASR_offine():
    global csv_original_scores, csv_found_words, subfolder_name
    global sliding_window_start_time_ms

    # init
    kws_asr_init()
    cfg.general.bool_output_csv = True
    csv_original_scores = []
    csv_found_words = []
    subfolder_name = ''

    # load wave
    wave_loader = WaveLoader(cfg.general.sample_rate)
    wave_loader.load_data(args.input_wav)
    wave_data = wave_loader.to_numpy()

    # sliding window
    windows_times = int((len(wave_data) - cfg.general.window_size_samples) * 1.0 / cfg.general.window_stride_samples) + 1
    for times in range(windows_times):

        # get audio data
        audio_data = wave_data[times * int(cfg.general.window_stride_samples): times * int(cfg.general.window_stride_samples) + int(cfg.general.window_size_samples)]
        print("[Information:] Audio data stram: {} - {}, length: {} ".format((times * int(cfg.general.window_stride_samples)), (times * int(cfg.general.window_stride_samples) + int(cfg.general.window_size_samples)), len(audio_data)))
        # print(audio_data)

        sliding_window_start_time_ms = (((times - 2) * int(cfg.general.window_stride_samples)) / cfg.general.sample_rate) * 1000
        run_kws_asr(audio_data)
    
    if cfg.general.bool_output_csv:
        csv_original_scores_pd = pd.DataFrame(csv_original_scores)
        csv_original_scores_pd.to_csv(os.path.join(args.output_folder, 'original_scores.csv'), index=False)
        csv_found_words_pd = pd.DataFrame(csv_found_words)
        csv_found_words_pd.to_csv(os.path.join(args.output_folder, 'found_words.csv'), index=False)


def KWS_ASR_offine_perfolder():
    global audio_data_container_np, feature_data_container_np
    global bool_weakup, counter_weakup, counter_asr
    global output_wave_list, output_kws_id
    global kws_container_np
    global csv_original_scores, csv_found_words, subfolder_name
    global sliding_window_start_time_ms

    # init
    kws_asr_init()
    cfg.general.bool_output_csv = True
    wave_list = os.listdir(args.input_folder)
    wave_list.sort()

    for idx in range(len(wave_list)):
        if not wave_list[idx].endswith('.wav'):
            continue
        
        # init
        audio_data_container_np = np.array([])
        feature_data_container_np = np.array([])
        kws_container_np = np.array([])         # kws 结构容器中，用于滑窗输出结果
        output_wave_list = []

        bool_weakup = False
        counter_weakup = 0
        counter_asr = 0
        output_kws_id = 1

        wave_path = os.path.join(args.input_folder, wave_list[idx])

        # init 
        csv_original_scores = []
        csv_found_words = []
        subfolder_name = os.path.basename(wave_path).split('.')[0]
    
        # mkdir
        output_path = os.path.join(args.output_folder, subfolder_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # load wave
        wave_loader = WaveLoader(cfg.general.sample_rate)
        wave_loader.load_data(wave_path)
        wave_data = wave_loader.to_numpy()

        # sliding window
        windows_times = int((len(wave_data) - cfg.general.window_size_samples) * 1.0 / cfg.general.window_stride_samples) + 1
        for times in range(windows_times):

            # get audio data
            audio_data = wave_data[times * int(cfg.general.window_stride_samples): times * int(cfg.general.window_stride_samples) + int(cfg.general.window_size_samples)]
            print("[Information:] Audio data stram: {} - {}, length: {} ".format((times * int(cfg.general.window_stride_samples)), (times * int(cfg.general.window_stride_samples) + int(cfg.general.window_size_samples)), len(audio_data)))
            # print(audio_data)

            sliding_window_start_time_ms = (((times - 2) * int(cfg.general.window_stride_samples)) / cfg.general.sample_rate) * 1000
            run_kws_asr(audio_data)
        
        if cfg.general.bool_output_csv:
            csv_original_scores_pd = pd.DataFrame(csv_original_scores)
            csv_original_scores_pd.to_csv(os.path.join(args.output_folder, subfolder_name, 'original_scores.csv'), index=False)
            csv_found_words_pd = pd.DataFrame(csv_found_words)
            csv_found_words_pd.to_csv(os.path.join(args.output_folder, subfolder_name, 'found_words.csv'), index=False)


if __name__ == "__main__":
    # 实现功能：语音唤醒 weakup 和关键词检索 asr 共同工作，目的是共用一套特征，节约资源
    # 方案一：实现 weakup + asr 
    # 方案二：在无 weakup 的情况下，实现 asr
    
    if args.test_mode == 0:
        KWS_ASR_offine()
    elif args.test_mode == 1:
        KWS_ASR_offine_perfolder()
        