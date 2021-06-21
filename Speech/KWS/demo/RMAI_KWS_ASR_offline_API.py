import argparse
from datetime import datetime
import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_feature_pyimpl import Feature
# from ASR.impl.asr_feature_cimpl import Feature
import ASR.impl.asr_decode_pyimpl as Decode_Python
import ASR.impl.asr_data_loader_pyimpl as WaveLoader_Python
# import ASR.impl.asr_data_loader_cimpl as WaveLoader_C

sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.file_tools import load_module_from_disk

caffe_root = "/home/huanyuan/code/caffe_ssd/"
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')
import caffe

# options 
# cfg = load_module_from_disk("/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_BWC_bpe.py")
# cfg = load_module_from_disk("/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_BWC_phoneme.py")
cfg = load_module_from_disk("/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_BWC_bpe_phoneme.py")
# cfg = load_module_from_disk("/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_MTA_XIAOAN.py")
# cfg = load_module_from_disk("/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_XIAORUI.py")
# cfg = load_module_from_disk("/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_MANDARIN_TAXI_3s.py")
# cfg = load_module_from_disk("/home/huanyuan/code/demo/Speech/KWS/demo/RMAI_KWS_ASR_options_MANDARIN_TAXI_4s.py")
cfg = cfg.cfg

# params
params_dict = {}
output_dict = {}

def param_init(bool_init_output_kws_id = True, subfolder_name=''):
    # container
    params_dict['audio_data_container_np'] = np.array([])
    params_dict['feature_data_container_np'] = np.array([])
    params_dict['kws_container_np'] = np.array([])         # kws 结构容器中，用于滑窗输出结果
    params_dict['output_wave_list'] = []

    params_dict['bool_weakup'] = False
    params_dict['counter_weakup'] = 0
    params_dict['counter_asr'] = cfg.general.asr_suppression_counter - 1

    output_dict['csv_original_scores'] = []
    output_dict['csv_found_words'] = []
    output_dict['subfolder_name'] = subfolder_name
    if bool_init_output_kws_id:
        output_dict['output_kws_id'] = 1
    output_dict['sliding_window_start_time_ms'] = 0

    # mkdir
    if cfg.general.bool_output_wave or cfg.general.bool_output_csv:
        if not os.path.exists(cfg.test.output_folder):
            os.makedirs(cfg.test.output_folder)

    
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


def kws_asr_init_normal():
    global kws_net
    global asr_net, decode_python

    # init model
    if cfg.general.bool_do_kws_weakup:
        kws_net = model_init(cfg.model.kws_prototxt_path, cfg.model.kws_model_path, cfg.model.kws_net_input_name, cfg.model.kws_chw_params.split(","), cfg.general.gpu)
    else:
        kws_net = None

    if cfg.general.bool_do_asr:
        asr_net = model_init(cfg.model.asr_prototxt_path, cfg.model.asr_model_path, cfg.model.asr_net_input_name, cfg.model.asr_chw_params.split(","), cfg.general.gpu)

        # init bpe dict 
        decode_python = Decode_Python.Decode()
        decode_python.init_symbol_list(cfg.model.asr_bpe)

        # init lm
        if cfg.general.decode_id == 1:
            decode_python.init_lm_model(cfg.model.lm_path)
    else:
        asr_net = None
        decode_python = None


def kws_asr_init_bpe_phoneme():
    global kws_net
    global asr_net_bpe, decode_python_bpe
    global asr_net_phoneme, decode_python_phoneme

    # init model
    if cfg.general.bool_do_kws_weakup:
        kws_net = model_init(cfg.model.kws_prototxt_path, cfg.model.kws_model_path, cfg.model.kws_net_input_name, cfg.model.kws_chw_params.split(","), cfg.general.gpu)
    else:
        kws_net = None

    if cfg.general.bool_do_asr:
        # init bpe net
        asr_net_bpe = model_init(cfg.model.asr_bpe_prototxt_path, cfg.model.asr_bpe_model_path, cfg.model.asr_bpe_net_input_name, cfg.model.asr_bpe_chw_params.split(","), cfg.general.gpu)

        # init bpe dict 
        decode_python_bpe = Decode_Python.Decode()
        decode_python_bpe.init_symbol_list(cfg.model.asr_bpe_dict)

        # init lm
        if cfg.general.bpe_decode_id == 1:
            decode_python_bpe.init_lm_model(cfg.model.asr_bpe_lm_path)

        # init phoneme net
        asr_net_phoneme = model_init(cfg.model.asr_phoneme_prototxt_path, cfg.model.asr_phoneme_model_path, cfg.model.asr_phoneme_net_input_name, cfg.model.asr_phoneme_chw_params.split(","), cfg.general.gpu)

        # init phoneme dict 
        decode_python_phoneme = Decode_Python.Decode()
        decode_python_phoneme.init_symbol_list(cfg.model.asr_phoneme_dict)

        # init lm
        if cfg.general.phoneme_decode_id == 1:
            decode_python_phoneme.init_lm_model(cfg.model.asr_phoneme_lm_path)

    else:
        asr_net_bpe = None
        decode_python_bpe = None

        asr_net_phoneme = None
        decode_python_phoneme = None


def kws_asr_init():
    if not cfg.general.asr_bpe_phoneme_on :
        kws_asr_init_normal()
    else:
        kws_asr_init_bpe_phoneme()


def papare_data_and_feature(audio_data):
    # 加载音频数据，用于打印输出
    if cfg.general.bool_output_wave:
        if len(params_dict['output_wave_list']) < cfg.general.total_time_samples:
            params_dict['output_wave_list'].extend(audio_data)
        else:
            params_dict['output_wave_list'] = params_dict['output_wave_list'][cfg.general.window_size_samples:]
            params_dict['output_wave_list'].extend(audio_data)

    # audio data
    # 拼接语音数据
    if len(params_dict['audio_data_container_np']):
        audio_data = np.concatenate((params_dict['audio_data_container_np'], audio_data), axis=0)
    
    # 存储指定时间的音频，用于后续拼接语音特征
    params_dict['audio_data_container_np'] = audio_data[len(audio_data) - cfg.general.window_container_samples:]
    # print("[Information:] Audio length: ", len(audio_data), len(params_dict['audio_data_container_np']))

    # feature
    # 计算特征
    feature = Feature(cfg.general.sample_rate, len(audio_data)/cfg.general.sample_rate, int(cfg.general.feature_freq), int(cfg.general.nfilt))
    feature.get_mel_int_feature(audio_data, len(audio_data))
    feature_data = feature.copy_mfsc_feature_int_to()
    # print("[Information:] Feature shape: ", feature_data.shape)

    # 拼接特征
    if not params_dict['feature_data_container_np'].shape[0]:
        params_dict['feature_data_container_np'] = feature_data
    elif params_dict['feature_data_container_np'].shape[0] < cfg.general.feature_container_time:
        params_dict['feature_data_container_np'] = np.concatenate((params_dict['feature_data_container_np'][: -cfg.general.feature_remove_after_time], feature_data), axis=0)
    else:
        params_dict['feature_data_container_np'] = np.concatenate((params_dict['feature_data_container_np'][cfg.general.feature_remove_before_time: -cfg.general.feature_remove_after_time], feature_data), axis=0)
    # print("[Information:] Feature container shape: ", params_dict['feature_data_container_np'].shape)


def output_wave(output_prefix_name):
    # save audio
    if cfg.general.bool_output_wave:
        # date_time = '-'.join('-'.join(str(datetime.now()).split('.')[0].split(' ')).split(':'))
        date_time = str(datetime.now()).split(' ')[0]
        output_path = os.path.join(cfg.test.output_folder, output_dict['subfolder_name'], '{}_{}_{}.wav'.format(output_prefix_name, date_time, output_dict['output_kws_id']))
        # output_path = os.path.join(cfg.test.output_folder, output_dict['subfolder_name'], '{}_starttime_{}.wav'.format(output_prefix_name, int(output_dict['sliding_window_start_time_ms'])))
        wave_loader = WaveLoader_Python.WaveLoader_Soundfile(cfg.general.sample_rate)
        # wave_loader = WaveLoader_Python.WaveLoader_Librosa(cfg.general.sample_rate)
        wave_loader.save_data(np.array(params_dict['output_wave_list']), output_path)
        output_dict['output_kws_id'] += 1


def run_kws():
    # 采用滑窗的方式判断是否触发 kws
    # 否则直接在 1s 内直接查找 kws，会漏掉起始和结尾点，造成漏唤醒
    # init
    kws_score_list = []

    if not cfg.general.bool_do_kws_weakup:
        return False, kws_score_list

    # 滑窗，模型前传
    # 每次送入 1s 数据，只需要对 1s 音频特征进行滑窗，模型前传；否则，会出现重复检测
    kws_weakup_times =  int((cfg.general.feature_time) * 1.0 / cfg.general.kws_stride_feature_time) + 1

    # 对每次送入的 1s 数据进行模型前传
    for kws_weakup_time in range(kws_weakup_times):
        end_feature_time = params_dict['feature_data_container_np'].shape[0] - (kws_weakup_times - kws_weakup_time) * cfg.general.kws_stride_feature_time
        start_feature_time = end_feature_time - int(cfg.general.kws_feature_time)
        assert start_feature_time >= 0, "kws weakup model 特征时间维度太大， 处理音频数据无法获得 {} 次滑窗结果".format(kws_weakup_times)
        # print("start_feature_time", start_feature_time, "end_feature_time", end_feature_time)

        feature_data_kws = params_dict['feature_data_container_np'][start_feature_time: end_feature_time,:]
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
    if len(params_dict['kws_container_np']):
        kws_score_np = np.concatenate((params_dict['kws_container_np'], kws_score_np), axis=0)
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
        params_dict['kws_container_np'] = np.zeros(np.array(kws_score_list).shape)
    else:
        # 存储一定时间的 kws 结果，用于后续滑窗获得结果
        params_dict['kws_container_np'] = np.array(kws_score_list)

    return bool_find_kws, kws_score_list


def run_asr_normal(contorl_kws_bool=True):
    if not cfg.general.bool_do_asr:
        return "cfg.general.bool_do_asr = False"

    # 获取特征
    feature_data_asr = params_dict['feature_data_container_np'].astype(np.float32)
    # print(feature_data_asr)

    # 模型前向传播
    asr_net.blobs[cfg.model.asr_net_input_name].data[...] = np.expand_dims(feature_data_asr, axis=0)
    net_output = asr_net.forward()[cfg.model.asr_net_output_name]
    net_output = np.squeeze(net_output)
    net_output = net_output.T
    # print(net_output.shape)
    # print(net_output)

    # # # debug: cpp & python 一致性测试
    # import cv2
    # # 保存矩阵
    # # cv_file = cv2.FileStorage("/home/huanyuan/share/audio_data/demo_test/test_python.xml", cv2.FILE_STORAGE_WRITE)
    # # cv_file.write('test', net_output)
    # # cv_file.release()

    # # 加载矩阵
    # # cv_file = cv2.FileStorage("/home/huanyuan/share/audio_data/demo_test/test_cpp.xml", cv2.FILE_STORAGE_READ)
    # # net_output = cv_file.getNode("test").mat()
    # # cv_file.release()
    # cv_file = cv2.FileStorage("/home/huanyuan/share/audio_data/demo_test/test_feature_cpp.xml", cv2.FILE_STORAGE_READ)
    # feature_data_fs = cv_file.getNode("feature").mat()
    # cv_file.release()
    # print("feature equal：", (feature_data_fs == feature_data_asr).all())
    
    # decode
    if cfg.general.decode_id == 0:
        decode_python.ctc_decoder(net_output)
    elif cfg.general.decode_id == 1:
        decode_python.beamsearch_decoder(net_output, 5, 0, bswt=1.0, lmwt=0.3)
    else:
        print("[Unknow:] cfg.general.decode_id. ")

    if cfg.general.language_id == 0:
        decode_python.show_result_id()
        decode_python.show_symbol()

        result_string = decode_python.output_symbol()
    elif cfg.general.language_id == 1:
        decode_python.show_result_id()
        decode_python.show_symbol()
        decode_python.show_symbol_english()

        # bpe
        decode_python.match_keywords_english_bpe(cfg.general.kws_list, cfg.general.kws_bpe_dict)
        result_string = decode_python.output_control_result_string(cfg.general.control_kws_list, contorl_kws_bool)
        
        # phoneme
        # # 鲁邦的匹配方式  
        # decode_python.match_keywords_english_phoneme_robust(cfg.general.kws_list, cfg.general.kws_phoneme_dict, cfg.general.control_kws_list, contorl_kws_bool)
        # # result_string = decode_python.output_result_string()
        # result_string = decode_python.output_control_result_string(cfg.general.control_kws_list, contorl_kws_bool)

        # # 严格匹配方式
        # decode_python.match_keywords_english_phoneme_strict(cfg.general.kws_list, cfg.general.kws_phoneme_dict, cfg.general.control_kws_list, contorl_kws_bool, cfg.general.kws_phoneme_param_dict)
        # # result_string = decode_python.output_result_string()
        # result_string = decode_python.output_control_result_string(cfg.general.control_kws_list, contorl_kws_bool)

        # # 自定义的匹配方式       
        # decode_python.match_keywords_english_phoneme_combine(cfg.general.kws_list, cfg.general.kws_phoneme_dict, cfg.general.control_kws_list, contorl_kws_bool, cfg.general.kws_phoneme_param_dict)
        # # result_string = decode_python.output_result_string()
        # result_string = decode_python.output_control_result_string(cfg.general.control_kws_list, contorl_kws_bool)

    else:
        print("[Unknow:] cfg.general.language_id. ")

    return result_string


def run_asr_bpe(contorl_kws_bool=True):
    if not cfg.general.bool_do_asr:
        return "cfg.general.bool_do_asr = False"

    # 获取特征
    feature_data_asr = params_dict['feature_data_container_np'].astype(np.float32)
    # print(feature_data_asr)

    # 模型前向传播：bpe
    asr_net_bpe.blobs[cfg.model.asr_bpe_net_input_name].data[...] = np.expand_dims(feature_data_asr, axis=0)
    net_output = asr_net_bpe.forward()[cfg.model.asr_bpe_net_output_name]
    net_output = np.squeeze(net_output)
    net_output = net_output.T
    # print(net_output.shape)
    # print(net_output)
    
    # decode：bpe
    if cfg.general.bpe_decode_id == 0:
        decode_python_bpe.ctc_decoder(net_output)
    elif cfg.general.bpe_decode_id == 1:
        decode_python_bpe.beamsearch_decoder(net_output, 5, 0, bswt=1.0, lmwt=0.3)
    else:
        print("[Unknow:] cfg.general.decode_id. ")

    if cfg.general.language_id == 0:
        decode_python_bpe.show_result_id()
        decode_python_bpe.show_symbol()

        result_string = decode_python_bpe.output_symbol()
    elif cfg.general.language_id == 1:
        decode_python_bpe.show_result_id()
        decode_python_bpe.show_symbol()
        decode_python_bpe.show_symbol_english()

        # bpe
        decode_python_bpe.match_keywords_english_bpe(cfg.general.kws_list, cfg.general.kws_bpe_dict)
        result_string = decode_python_bpe.output_control_result_string(cfg.general.control_kws_list, contorl_kws_bool)
    else:
        print("[Unknow:] cfg.general.language_id. ")

    return result_string


def run_asr_phoneme(contorl_kws_bool=True):
    
    if not cfg.general.bool_do_asr:
        return "cfg.general.bool_do_asr = False"

    # 获取特征
    feature_data_asr = params_dict['feature_data_container_np'].astype(np.float32)
    # print(feature_data_asr)

    # 模型前向传播：phoneme
    asr_net_phoneme.blobs[cfg.model.asr_phoneme_net_input_name].data[...] = np.expand_dims(feature_data_asr, axis=0)
    net_output = asr_net_phoneme.forward()[cfg.model.asr_phoneme_net_output_name]
    net_output = np.squeeze(net_output)
    net_output = net_output.T
    # print(net_output.shape)
    # print(net_output)
    
    # decode：phoneme
    if cfg.general.phoneme_decode_id == 0:
        decode_python_phoneme.ctc_decoder(net_output)
    elif cfg.general.phoneme_decode_id == 1:
        decode_python_phoneme.beamsearch_decoder(net_output, 5, 0, bswt=1.0, lmwt=0.3)
    else:
        print("[Unknow:] cfg.general.decode_id. ")

    if cfg.general.language_id == 0:
        decode_python_phoneme.show_result_id()
        decode_python_phoneme.show_symbol()

        result_string = decode_python_phoneme.output_symbol()
    elif cfg.general.language_id == 1:
        decode_python_phoneme.show_result_id()
        decode_python_phoneme.show_symbol()
        decode_python_phoneme.show_symbol_english()

        # phoneme
        # # 鲁邦的匹配方式  
        # decode_python_phoneme.match_keywords_english_phoneme_robust(cfg.general.kws_list, cfg.general.kws_phoneme_dict, cfg.general.control_kws_list, contorl_kws_bool)
        # # result_string = decode_python_phoneme.output_result_string()
        # result_string = decode_python_phoneme.output_control_result_string(cfg.general.control_kws_list, contorl_kws_bool)

        # 严格匹配方式
        decode_python_phoneme.match_keywords_english_phoneme_strict(cfg.general.kws_list, cfg.general.kws_phoneme_dict, cfg.general.control_kws_list, contorl_kws_bool, cfg.general.kws_phoneme_param_dict)
        # result_string = decode_python_phoneme.output_result_string()
        result_string = decode_python_phoneme.output_control_result_string(cfg.general.control_kws_list, contorl_kws_bool)

        # # 自定义的匹配方式       
        # decode_python_phoneme.match_keywords_english_phoneme_combine(cfg.general.kws_list, cfg.general.kws_phoneme_dict, cfg.general.control_kws_list, contorl_kws_bool, cfg.general.kws_phoneme_param_dict)
        # # result_string = decode_python_phoneme.output_result_string()
        # result_string = decode_python_phoneme.output_control_result_string(cfg.general.control_kws_list, contorl_kws_bool)
    else:
        print("[Unknow:] cfg.general.language_id. ")

    return result_string


def run_asr_bpe_phoneme(contorl_kws_bool=True):

    result_string = run_asr_bpe(contorl_kws_bool)

    if len(result_string) and result_string != "cfg.general.bool_do_asr = False":
        if contorl_kws_bool:
            print("Bpe Detect Command: ", result_string)
        else:
            print("Bpe Detect Command: ", result_string)
            print("Followed Phoneme Detect: ")
            
            result_string = run_asr_phoneme(contorl_kws_bool)
    return result_string


def run_asr(contorl_kws_bool=True):

    if not cfg.general.asr_bpe_phoneme_on:
        return run_asr_normal(contorl_kws_bool)
    else:
        return run_asr_bpe_phoneme(contorl_kws_bool)


def run_kws_asr(audio_data):
    # 准备数据和特征
    papare_data_and_feature(audio_data)

    # 如果语音特征未装满容器，不进行唤醒和关键词检测
    if params_dict['feature_data_container_np'].shape[0] < cfg.general.feature_container_time:
        return

    # 方案一：进行 kws 唤醒词检测，若检测到唤醒词，未来三秒进行 asr 检测
    # kws
    if not params_dict['bool_weakup']:
        bool_find_kws, kws_score_list = run_kws()

        if bool_find_kws:
            # 打印结果
            print("\n===============!!!!!!!!!!!!!!===============")
            print("********************************************")
            print("** ")
            print("** [Information:] Device Weakup: ", "Weakup")
            print("** ")
            print("********************************************\n")

            params_dict['bool_weakup'] = True
            
            # save audio
            output_wave("Weakup")

            if cfg.general.bool_output_csv:
                output_dict['csv_found_words'].append({'label':"Weakup", 'start_time':int(output_dict['sliding_window_start_time_ms']), 'end_time': int(output_dict['sliding_window_start_time_ms'] + cfg.general.total_time_ms)})

        if cfg.general.bool_output_csv:
            for idx in range(len(kws_score_list)):
                output_dict['csv_original_scores'].append({'start_time':output_dict['sliding_window_start_time_ms'] + idx * cfg.general.kws_stride_feature_time * 10, 'score':",".join([str(kws_score_list[idx][idy]) for idy in range(kws_score_list[idx].shape[0])])})

    else:
        params_dict['counter_weakup'] += 1
        if params_dict['counter_weakup'] >= cfg.general.kws_suppression_counter:
            params_dict['counter_weakup'] = 0
            params_dict['bool_weakup'] = False
            # 控制 asr 的间隔时间
            params_dict['counter_asr'] -= 1

            # asr
            result_string = run_asr(True)
        
            # 打印结果
            if len(result_string) and result_string != "cfg.general.bool_do_asr = False":
                print("\n===============!!!!!!!!!!!!!!===============")
                print("********************************************")
                print("** ")
                print("** [Information:] Detect Command: ", result_string)
                print("** ")
                print("********************************************\n")

                # save audio
                output_wave("ASR_" + result_string)
            else:
                print("\n** [Information:] Detecting ... ")
        
        # if cfg.general.bool_output_csv:
        #     _, kws_score_list = run_kws()
        #     for idx in range(len(kws_score_list)):
        #         output_dict['csv_original_scores'].append({'start_time':output_dict['sliding_window_start_time_ms'] + idx * cfg.general.kws_stride_feature_time * 10, 'score':",".join([str(kws_score_list[idx][idy]) for idy in range(kws_score_list[idx].shape[0])])})

    # 方案二：进行 asr 检测，间隔一定时长
    # asr
    # 如果检测到唤醒词，则执行方案一
    if params_dict['bool_weakup']:
        params_dict['counter_asr'] = 0
    else:
        params_dict['counter_asr'] += 1

    if params_dict['counter_asr'] == cfg.general.asr_suppression_counter:
        params_dict['counter_asr'] = 0

        result_string = run_asr(False)

        # 打印结果
        if len(result_string) and result_string != "cfg.general.bool_do_asr = False":
            print("\n===============!!!!!!!!!!!!!!===============")
            print("********************************************")
            print("** ")
            print("** [Information:] Detect Command: ", result_string)
            print("** ")
            print("********************************************\n")
            # save audio
            output_wave("ASR_" + result_string)
        else:
            print("\n** [Information:] Detecting ... ")


def KWS_ASR_offine():
    # param_init
    param_init()

    # kws_asr_init
    kws_asr_init()

    # load wave
    # wave_loader = WaveLoader_C.WaveLoader(cfg.general.sample_rate)
    wave_loader = WaveLoader_Python.WaveLoader_Soundfile(cfg.general.sample_rate)
    # wave_loader = WaveLoader_Python.WaveLoader_Librosa(cfg.general.sample_rate)
    wave_loader.load_data(cfg.test.input_wav)
    wave_data = wave_loader.to_numpy()

    # sliding window
    windows_times = int((len(wave_data) - cfg.general.window_size_samples) * 1.0 / cfg.general.window_stride_samples) + 1
    for times in range(windows_times):

        # get audio data
        audio_data = wave_data[times * int(cfg.general.window_stride_samples): times * int(cfg.general.window_stride_samples) + int(cfg.general.window_size_samples)]
        print("[Information:] Audio data stream: {} - {}, length: {} ".format((times * int(cfg.general.window_stride_samples)), (times * int(cfg.general.window_stride_samples) + int(cfg.general.window_size_samples)), len(audio_data)))
        print(audio_data)

        output_dict['sliding_window_start_time_ms'] = (((times - 2) * int(cfg.general.window_stride_samples)) / cfg.general.sample_rate) * 1000
        run_kws_asr(audio_data)
    
    if cfg.general.bool_output_csv:
        csv_original_scores_pd = pd.DataFrame(output_dict['csv_original_scores'])
        csv_original_scores_pd.to_csv(os.path.join(cfg.test.output_folder, 'original_scores.csv'), index=False)
        csv_found_words_pd = pd.DataFrame(output_dict['csv_found_words'])
        csv_found_words_pd.to_csv(os.path.join(cfg.test.output_folder, 'found_words.csv'), index=False)


def KWS_ASR_offine_perfolder():
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
        param_init(bool_init_output_kws_id = False, subfolder_name='')
        # param_init(bool_init_output_kws_id = False, subfolder_name=os.path.basename(wave_path).split('.')[0])

        wave_path = os.path.join(cfg.test.input_folder, wave_list[idx])
        print("[Information:] Audio path: ", wave_path)
    
        # mkdir
        if cfg.general.bool_output_wave or cfg.general.bool_output_csv:
            output_path = os.path.join(cfg.test.output_folder, output_dict['subfolder_name'])
            if not os.path.exists(output_path):
                os.makedirs(output_path)

        # load wave
        # wave_loader = WaveLoader_C.WaveLoader(cfg.general.sample_rate)
        wave_loader = WaveLoader_Python.WaveLoader_Soundfile(cfg.general.sample_rate)
        # wave_loader = WaveLoader_Python.WaveLoader_Librosa(cfg.general.sample_rate)
        wave_loader.load_data(wave_path)
        wave_data = wave_loader.to_numpy()

        # sliding window
        windows_times = int((len(wave_data) - cfg.general.window_size_samples) * 1.0 / cfg.general.window_stride_samples) + 1
        for times in range(windows_times):

            # get audio data
            audio_data = wave_data[times * int(cfg.general.window_stride_samples): times * int(cfg.general.window_stride_samples) + int(cfg.general.window_size_samples)]
            print("[Information:] Audio data stram: {} - {}, length: {} ".format((times * int(cfg.general.window_stride_samples)), (times * int(cfg.general.window_stride_samples) + int(cfg.general.window_size_samples)), len(audio_data)))
            print(audio_data)

            output_dict['sliding_window_start_time_ms'] = (((times - 2) * int(cfg.general.window_stride_samples)) / cfg.general.sample_rate) * 1000
            run_kws_asr(audio_data)
        
        if cfg.general.bool_output_csv:
            csv_original_scores_pd = pd.DataFrame(output_dict['csv_original_scores'])
            csv_original_scores_pd.to_csv(os.path.join(cfg.test.output_folder, output_dict['subfolder_name'], 'original_scores.csv'), index=False)
            csv_found_words_pd = pd.DataFrame(output_dict['csv_found_words'])
            csv_found_words_pd.to_csv(os.path.join(cfg.test.output_folder, output_dict['subfolder_name'], 'found_words.csv'), index=False)


if __name__ == "__main__":
    # 实现功能：语音唤醒 weakup 和关键词检索 asr 共同工作，目的是共用一套特征，节约资源
    # 方案一：实现 weakup + asr 
    # 方案二：在无 weakup 的情况下，实现 asr
    
    if cfg.test.test_mode == 0:
        KWS_ASR_offine()
    elif cfg.test.test_mode == 1:
        KWS_ASR_offine_perfolder()
        