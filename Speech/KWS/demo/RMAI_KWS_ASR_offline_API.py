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

sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.file_tools import load_module_from_disk

caffe_root = "/home/huanyuan/code/caffe_ssd/"
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')
import caffe

# options 
# cfg = load_module_from_disk("/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_BWC.py")
cfg = load_module_from_disk("/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_MTA_XIAOAN.py")
# cfg = load_module_from_disk("/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_XIAORUI.py")
cfg = cfg.cfg


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
    if cfg.general.bool_do_kws_weakup:
        kws_net = model_init(cfg.model.kws_prototxt_path, cfg.model.kws_model_path, cfg.model.kws_net_input_name, cfg.model.kws_chw_params.split(","), cfg.general.gpu)
    else:
        kws_net = None

    if cfg.general.bool_do_asr:
        asr_net = model_init(cfg.model.asr_prototxt_path, cfg.model.asr_model_path, cfg.model.asr_net_input_name, cfg.model.asr_chw_params.split(","), cfg.general.gpu)
    else:
        asr_net = None

    # init bpe
    decode_python = Decode_Python.Decode()
    decode_python.init_ast_symbol_list(cfg.model.asr_bpe)

    # mkdir
    if not os.path.exists(cfg.test.output_folder):
        os.makedirs(cfg.test.output_folder)
    

def param_init():
    global audio_data_container_np, feature_data_container_np
    global bool_weakup, counter_weakup, counter_asr
    global output_wave_list, output_kws_id
    global kws_container_np

    # container
    audio_data_container_np = np.array([])
    feature_data_container_np = np.array([])
    kws_container_np = np.array([])         # kws 结构容器中，用于滑窗输出结果
    output_wave_list = []

    bool_weakup = False
    counter_weakup = 0
    counter_asr = 0
    output_kws_id = 1


def run_kws():
    # 采用滑窗的方式判断是否触发 kws
    # 否则直接在 1s 内直接查找 kws，会漏掉起始和结尾点，造成漏唤醒
    global kws_container_np

    # init
    kws_score_list = []

    if not cfg.general.bool_do_kws_weakup:
        return False, kws_score_list

    # 滑窗，模型前传
    # 每次送入 1s 数据，只需要对 1s 音频特征进行滑窗，模型前传；否则，会出现重复检测
    kws_weakup_times =  int((cfg.general.feature_time) * 1.0 / cfg.general.kws_stride_feature_time) + 1

    # 对每次送入的 1s 数据进行模型前传
    for kws_weakup_time in range(kws_weakup_times):
        end_feature_time = feature_data_container_np.shape[0] - (kws_weakup_times - kws_weakup_time) * cfg.general.kws_stride_feature_time
        start_feature_time = end_feature_time - int(cfg.general.kws_feature_time)
        assert start_feature_time >= 0, "kws weakup model 特征时间维度太大， 处理音频数据无法获得 {} 次滑窗结果".format(kws_weakup_times)
        # print("start_feature_time", start_feature_time, "end_feature_time", end_feature_time)

        feature_data_kws = feature_data_container_np[start_feature_time: end_feature_time,:]
        feature_data_kws = feature_data_kws.astype(np.float32)
        if cfg.model.kws_transpose:
            feature_data_kws = feature_data_kws.T
        kws_net.blobs[cfg.model.kws_net_input_name].data[...] = np.expand_dims(feature_data_kws, axis=0)

        net_output = kws_net.forward()[cfg.model.kws_net_output_name]
        net_output = np.squeeze(net_output)
        kws_score_list.append(net_output.copy())
        # print(feature_data_kws.shape, net_output)
        # print(net_output)

    # 如果有保留的 kws 结果，进行拼接
    kws_score_np = np.array(kws_score_list)
    if len(kws_container_np):
        kws_score_np = np.concatenate((kws_container_np, kws_score_np), axis=0)
    # print(kws_score_np, kws_score_np.shape)

    bool_find_kws = False
    for kws_idx in range(len(kws_score_np) + 1 - kws_weakup_times):
        # 滑窗，获得后处理结果
        detected_number = 0 
        for kws_times in range(kws_weakup_times):
            if kws_score_np[kws_idx + kws_times][-1] > cfg.general.kws_detection_threshold:
                detected_number += 1

        if detected_number >= kws_weakup_times * cfg.general.kws_detection_number_threshold:
            bool_find_kws = True
    
    if bool_find_kws:
        kws_container_np = np.zeros(np.array(kws_score_list).shape)
    else:
        # 存储一定时间的 kws 结果，用于后续滑窗获得结果
        kws_container_np = np.array(kws_score_list)

    return bool_find_kws, kws_score_list


def run_asr():
    if not cfg.general.bool_do_asr:
        return "cfg.general.bool_do_asr = False", "", ""

    feature_data_asr = feature_data_container_np.astype(np.float32)
    asr_net.blobs[cfg.model.asr_net_input_name].data[...] = np.expand_dims(feature_data_asr, axis=0)

    net_output = asr_net.forward()[cfg.model.asr_net_output_name]
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
                output_path = os.path.join(cfg.test.output_folder, subfolder_name, 'label_{}_starttime_{}.wav'.format(cfg.model.kws_label, int(sliding_window_start_time_ms)))
                wave_loader = WaveLoader(cfg.general.sample_rate)
                wave_loader.save_data(np.array(output_wave_list), output_path)

                csv_found_words.append({'label':cfg.model.kws_label, 'start_time':int(sliding_window_start_time_ms), 'end_time': int(sliding_window_start_time_ms + cfg.general.total_time_ms)})
                
            elif cfg.general.bool_output_wave:
                wave_loader = WaveLoader(cfg.general.sample_rate)
                date_time = '-'.join('-'.join(str(datetime.now()).split('.')[0].split(' ')).split(':'))
                wave_loader.save_data(np.array(output_wave_list), os.path.join(cfg.test.output_folder, "Weakup_{}_{:0>4d}.wav".format(date_time, output_kws_id)))
                output_kws_id += 1

        if cfg.general.bool_output_csv:
            for idx in range(len(kws_score_list)):
                csv_original_scores.append({'start_time':sliding_window_start_time_ms + idx * cfg.general.kws_stride_feature_time * 10, 'score':",".join([str(kws_score_list[idx][idy]) for idy in range(kws_score_list[idx].shape[0])])})

    else:
        counter_weakup += 1
        if counter_weakup >= cfg.general.kws_suppression_counter:
            counter_weakup = 0
            bool_weakup = False
            # 控制 asr 的间隔时间
            counter_asr -= 1

            # asr
            result_string, control_command_string, _ = run_asr()
            print("[Information:] kws asr outKeyword: ", result_string)
            # print("[Information:] kws asr outKeyword: ", control_command_string)
        
        # if cfg.general.bool_output_csv:
        #     _, kws_score_list = run_kws()
        #     for idx in range(len(kws_score_list)):
        #         csv_original_scores.append({'start_time':sliding_window_start_time_ms + idx * cfg.general.kws_stride_feature_time * 10, 'score':",".join([str(kws_score_list[idx][idy]) for idy in range(kws_score_list[idx].shape[0])])})

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

    # params init 
    cfg.general.bool_output_csv = True
    
    # param_init
    param_init()

    # kws_asr_init
    kws_asr_init()

    csv_original_scores = []
    csv_found_words = []
    subfolder_name = ''

    # load wave
    wave_loader = WaveLoader(cfg.general.sample_rate)
    wave_loader.load_data(cfg.test.input_wav)
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
        csv_original_scores_pd.to_csv(os.path.join(cfg.test.output_folder, 'original_scores.csv'), index=False)
        csv_found_words_pd = pd.DataFrame(csv_found_words)
        csv_found_words_pd.to_csv(os.path.join(cfg.test.output_folder, 'found_words.csv'), index=False)


def KWS_ASR_offine_perfolder():
    global csv_original_scores, csv_found_words, subfolder_name
    global sliding_window_start_time_ms

    # params init 
    cfg.general.bool_output_csv = True

    # param_init
    param_init()

    # kws_asr_init
    kws_asr_init()

    wave_list = os.listdir(cfg.test.input_folder)
    wave_list.sort()

    for idx in range(len(wave_list)):
        if not wave_list[idx].endswith('.wav'):
            continue
        
        # param_init
        param_init()

        wave_path = os.path.join(cfg.test.input_folder, wave_list[idx])

        # init 
        csv_original_scores = []
        csv_found_words = []
        subfolder_name = os.path.basename(wave_path).split('.')[0]
    
        # mkdir
        output_path = os.path.join(cfg.test.output_folder, subfolder_name)
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
            csv_original_scores_pd.to_csv(os.path.join(cfg.test.output_folder, subfolder_name, 'original_scores.csv'), index=False)
            csv_found_words_pd = pd.DataFrame(csv_found_words)
            csv_found_words_pd.to_csv(os.path.join(cfg.test.output_folder, subfolder_name, 'found_words.csv'), index=False)


if __name__ == "__main__":
    # 实现功能：语音唤醒 weakup 和关键词检索 asr 共同工作，目的是共用一套特征，节约资源
    # 方案一：实现 weakup + asr 
    # 方案二：在无 weakup 的情况下，实现 asr
    
    if cfg.test.test_mode == 0:
        KWS_ASR_offine()
    elif cfg.test.test_mode == 1:
        KWS_ASR_offine_perfolder()
        