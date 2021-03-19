import argparse
from datetime import datetime
import numpy as np
import os
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_data_loader_pyimpl import WaveLoader
from ASR.impl.asr_feature_pyimpl import Feature
import ASR.impl.asr_decode_cimpl as Decode_C
import ASR.impl.asr_decode_pyimpl as Decode_Python

caffe_root = "/home/huanyuan/code/caffe_ssd/"
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')
import caffe


# params
sample_rate = 16000
window_size_ms = 1000                   # 每次送入 1s 数据
window_stride_ms = 1000                 # 每次间隔 1s 时间

total_time_ms = 3000
feature_freq = 64
feature_time = 296
kws_feature_time = 196
kws_stride_feature_time = 10            # kws 每间隔 10 个 feature_time 进行一次检索
kws_detection_threshold = 0.8           # kws 检测阈值 0.8
kws_detection_number_threshold = 0.5    # kws 计数阈值 0.5
kws_suppression_counter = 3             # kws 激活后抑制时间 3s

asr_feature_time = 296
asr_suppression_counter = 2             # asr 激活后抑制时间，间隔 2s 执行一次 asr 检测

data_container_ms = 100                 # 语音数据容器中，装有音频数据 100 ms
feature_container_time = 296            # 语音特征容器中，装有时间维度 296
feature_remove_after_time = 6           # 为保证特征一致，拼接特征需要丢弃最后的时间维度 6
feature_remove_before_time = 100        # 为保证特征一致，拼接特征需要丢弃之前的时间维度 100

# init 
window_size_samples = int(sample_rate * window_size_ms / 1000)
window_stride_samples = int(sample_rate * window_stride_ms / 1000)
window_container_samples = int(sample_rate * data_container_ms / 1000)
total_time_samples = int(sample_rate * total_time_ms / 1000)

# container
output_audio_list = []
feature_data_container_np = np.array([])
audio_data_container_np = np.array([])

bool_weakup = False
counter_weakup = 0

counter_asr = 0

# argparse
default_kws_model_path = "/mnt/huanyuan/model/audio_model/caffe_model/kws_xiaorui_res15/res15_03162011.caffemodel"
default_kws_prototxt_path = "/mnt/huanyuan/model/audio_model/caffe_model/kws_xiaorui_res15/res15_03162011.prototxt"
default_kws_net_input_name = "blob1"
default_kws_net_output_name = "Softmax"
default_kws_chw_params = "1,196,64"

default_asr_model_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/english_0202_better.caffemodel"
default_asr_prototxt_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/english_0202_mark.prototxt"
default_asr_net_input_name = "data"
default_asr_net_output_name = "conv39"
default_asr_chw_params = "1,296,64"
default_asr_bpe = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/english_bpe.txt"

# default_input_wav = "/home/huanyuan/share/audio_data/english_wav/1-0127-asr_16k.wav"
# default_input_wav = "/mnt/huanyuan/model/test_straming_wav/xiaorui_12162020_training_60_001.wav"
default_input_wav = "/mnt/huanyuan/data/speech/Recording_sample/iphone/test-kws-asr.wav"
default_output_folder = "/mnt/huanyuan/data/speech/Recording_sample/demo_kws_asr_online_api/{}".format(datetime.now())
default_gpu = True

parser = argparse.ArgumentParser(description='Streamax KWS ASR offine Engine')
parser.add_argument('--input_wav', type=str, default=default_input_wav)
parser.add_argument('--kws_model_path', type=str, default=default_kws_model_path)
parser.add_argument('--kws_prototxt_path', type=str, default=default_kws_prototxt_path)
parser.add_argument('--kws_net_input_name', type=str, default=default_kws_net_input_name)
parser.add_argument('--kws_net_output_name', type=str, default=default_kws_net_output_name)
parser.add_argument('--kws_chw_params', type=str, default=default_kws_chw_params)
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


def run_kws():
    kws_stride_times = int((feature_data_container_np.shape[0] - kws_feature_time) * 1.0 / kws_stride_feature_time) + 1

    detected_number = 0
    for times in range(kws_stride_times):
        feature_data_kws = feature_data_container_np[times * int(kws_stride_feature_time): times * int(kws_stride_feature_time) + int(kws_feature_time),:]
        feature_data_kws = feature_data_kws.astype(np.float32)
        kws_net.blobs[args.kws_net_input_name].data[...] = np.expand_dims(feature_data_kws, axis=0)

        net_output = kws_net.forward()[args.kws_net_output_name]
        net_output = np.squeeze(net_output)
        # print(times, feature_data_kws.shape, net_output)

        if net_output[-1] > kws_detection_threshold:
            detected_number += 1

    if detected_number >= kws_stride_times * kws_detection_number_threshold:
        return True
    return False


def run_asr():
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
    global output_audio_list, output_audio_time, output_kws_id

    # 加载音频数据，用于打印输出
    if len(output_audio_list) < total_time_samples:
        output_audio_list.extend(audio_data)
    else:
        output_audio_list = output_audio_list[window_size_samples:]
        output_audio_list.extend(audio_data)

    # 如果有保留的音频数据，进行拼接
    if len(audio_data_container_np):
        audio_data = np.concatenate((audio_data_container_np, audio_data), axis=0)
    
    # 存储一定时间的音频，用于后续计算特征
    audio_data_container_np = audio_data[len(audio_data) - window_container_samples:]
    # print("[Information:] Audio length: ", len(audio_data), len(audio_data_container_np))

    # 计算特征
    feature = Feature(len(audio_data), int(feature_freq))
    feature.get_mel_int_feature(audio_data, len(audio_data))
    feature_data = feature.copy_mfsc_feature_int_to()
    # print("[Information:] Feature shape: ", feature_data.shape)

    # 更新特征
    if not feature_data_container_np.shape[0]:
        feature_data_container_np = feature_data
    elif feature_data_container_np.shape[0] < feature_container_time:
        feature_data_container_np = np.concatenate((feature_data_container_np[: -feature_remove_after_time], feature_data), axis=0)
    else:
        feature_data_container_np = np.concatenate((feature_data_container_np[feature_remove_before_time: -feature_remove_after_time], feature_data), axis=0)
    # print("[Information:] Feature container shape: ", feature_data_container_np.shape)

    # 如果语音特征未装满容器，不进行唤醒和关键词检测
    if feature_data_container_np.shape[0] < feature_container_time:
        return

    # 方案一：先进行 kws 唤醒词检测，若检测到唤醒词，未来三秒进行 asr 检测
    # kws
    if not bool_weakup:
        bool_find_kws = run_kws()

        if bool_find_kws:
            print("[Information:] Find Kws Weakup")
            bool_weakup = True

            # save audio
            wave_loader = WaveLoader(sample_rate)
            wave_loader.save_data(np.array(output_audio_list), os.path.join(args.output_folder, "kws_{:0>4d}.wav".format(output_kws_id)))
            # wave_loader.save_data(np.array(output_audio_list), os.path.join(args.output_folder, "kws_start_time_{}.wav".format(output_audio_time)))
            output_kws_id += 1
    else:
        counter_weakup += 1
        if counter_weakup == kws_suppression_counter:
            counter_weakup = 0
            bool_weakup = False
            # 控制 asr 的间隔时间
            counter_asr -= 1

            # asr
            result_string, control_command_string, _ = run_asr()
            print("[Information:] kws asr outKeyword: ", result_string)
            # print("[Information:] kws asr outKeyword: ", control_command_string)
            # if result_string != "":
            # wave_loader = WaveLoader(sample_rate)
            # wave_loader.save_data(np.array(output_audio_list), 
            #                     os.path.join(args.output_folder, "kws_asr_start_time_{}_{}.wav".format(output_audio_time, "_".join(result_string.split(" ")))))

    # 方案二：间隔一定时长进行 asr 检测
    # asr
    # 如果检测到唤醒词，则执行方案一
    if bool_weakup:
        counter_asr = 0
    else:
        counter_asr += 1

    if counter_asr == asr_suppression_counter:
        counter_asr = 0

        result_string, _, not_control_command_string = run_asr()
        print("[Information:] asr outKeyword: ", result_string)
        # print("[Information:] asr outKeyword: ", not_control_command_string)
        # if result_string != "":
        # wave_loader = WaveLoader(sample_rate)
        # wave_loader.save_data(np.array(output_audio_list), 
        #                     os.path.join(args.output_folder, "asr_start_time_{}_{}.wav".format(output_audio_time, "_".join(result_string.split(" ")))))


def kws_asr_init():
    global kws_net, asr_net
    global decode_python
    global output_kws_id

    # init model
    kws_net = model_init(args.kws_prototxt_path, args.kws_model_path, args.kws_net_input_name, args.kws_chw_params.split(","), args.gpu)
    asr_net = model_init(args.asr_prototxt_path, args.asr_model_path, args.asr_net_input_name, args.asr_chw_params.split(","), args.gpu)

    # init bpe
    decode_python = Decode_Python.Decode()
    decode_python.init_ast_symbol_list(args.asr_bpe)

    # mkdir
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # init param
    output_kws_id = 1


def KWS_ASR_offine():
    global output_audio_time

    # init
    kws_asr_init()

    # load wave
    wave_loader = WaveLoader(sample_rate)
    wave_loader.load_data(args.input_wav)
    wave_data = wave_loader.to_numpy()

    # sliding window
    windows_times = int((len(wave_data) - window_size_samples) * 1.0 / window_stride_samples) + 1
    for times in range(windows_times):

        # get audio data
        audio_data = wave_data[times * int(window_stride_samples): times * int(window_stride_samples) + int(window_size_samples)]
        # print("[Information:] Audio data stram: {} - {}, length: {} ".format((times * int(window_stride_samples))/sample_rate, (times * int(window_stride_samples) + int(window_size_samples))/sample_rate, len(audio_data)))
        print("[Information:] Audio data stram: {} - {}, length: {} ".format((times * int(window_stride_samples)), (times * int(window_stride_samples) + int(window_size_samples)), len(audio_data)))

        output_audio_time = ((times - 2) * int(window_stride_samples)) / sample_rate
        run_kws_asr(audio_data)


if __name__ == "__main__":
    # 实现功能：语音唤醒 weakup 和关键词检索 kws 共同工作，目的是共用一套特征，节约资源
    # 方案一：实现 weakup + kws 
    # 方案二：在无 weakup 的情况下，实现 kws
    
    KWS_ASR_offine()