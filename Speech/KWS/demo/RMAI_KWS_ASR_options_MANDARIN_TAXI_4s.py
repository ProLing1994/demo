import numpy as np

from datetime import datetime
from easydict import EasyDict as edict


__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

__C.general.window_size_ms = 1000                   # 每次送入 1s 数据
__C.general.window_stride_ms = 1000                 # 每次间隔 1s 时间
__C.general.total_time_ms = 4000                    # 算法处理时长 4s 时间

__C.general.sample_rate = 16000
__C.general.nfilt = 64                              # 计算特征中，Mel 滤波器个数
__C.general.feature_freq = 64                       # 计算特征维度
__C.general.feature_time = 96                       # 每次送入 1s 数据，对应的特征时间维度 96

# kws
# activate bwc
__C.general.kws_feature_time = 192                  # kws 网络特征时间维度
__C.general.kws_stride_feature_time = 10            # kws 每间隔 10 个 feature_time 进行一次检索, 对应滑窗 100 ms，共检测 10 次
__C.general.kws_detection_threshold = 0.5           # kws 检测阈值 0.5
__C.general.kws_detection_number_threshold = 0.5    # kws 计数阈值 0.5
__C.general.kws_suppression_counter = 4             # kws 激活后抑制时间 4s

# asr mandarin taxi
__C.general.language_id = 0			                # 0： chinese  1： english
__C.general.decode_id = 1			                # 0： greedy  1： beamsearch
__C.general.asr_feature_time = 396                  # asr 网络特征时间维度，与语音特征容器长度相同
# __C.general.asr_suppression_counter = 3             # asr 激活后抑制时间，间隔 3s 执行一次 asr 检测
__C.general.asr_suppression_counter = 2             # asr 激活后抑制时间，间隔 2s 执行一次 asr 检测
__C.general.asr_bpe_phoneme_on = False               # asr 使用 bpe 和 phoneme 两个 model

# container
__C.general.audio_container_ms = 100                # 语音数据容器中，装有音频数据 100 ms
__C.general.audio_container_time = 10               # 语音数据容器中，装有音频数据 100 ms，对应特征维度 10
__C.general.feature_container_time = 396            # 语音特征容器中，装有时间维度 296
__C.general.feature_remove_after_time = 6           # 为保证特征一致，拼接特征需要丢弃最后的时间维度 6
__C.general.feature_remove_before_time = 100        # 为保证特征一致，拼接特征需要丢弃之前的时间维度 100

# on-off
# __C.general.bool_do_kws_weakup = True
__C.general.bool_do_kws_weakup = False
__C.general.bool_do_asr = True
# __C.general.bool_output_wave = True
__C.general.bool_output_wave = False
__C.general.bool_output_csv = False
__C.general.gpu = True

# init 
__C.general.window_size_samples = int(__C.general.sample_rate * __C.general.window_size_ms / 1000)
__C.general.window_stride_samples = int(__C.general.sample_rate * __C.general.window_stride_ms / 1000)
__C.general.window_container_samples = int(__C.general.sample_rate * __C.general.audio_container_ms / 1000)
__C.general.total_time_samples = int(__C.general.sample_rate * __C.general.total_time_ms / 1000)


##################################
# model parameters
##################################

__C.model = {}

# kws
# xiaorui
__C.model.kws_model_path = "/mnt/huanyuan/model/audio_model/hisi_model/kws_xiaorui16k_tc_resnet14/xiaorui16k_tc-resnet14_hisi_6_1_checkpoints_1999/kws_xiaorui16k_tc_resnet14_hisi_6_1_05312021.caffemodel"
__C.model.kws_prototxt_path = "/mnt/huanyuan/model/audio_model/hisi_model/kws_xiaorui16k_tc_resnet14/xiaorui16k_tc-resnet14_hisi_6_1_checkpoints_1999/kws_xiaorui16k_tc_resnet14_hisi_6_1_05312021.prototxt"
__C.model.kws_label = "xiaorui"
__C.model.kws_net_input_name = "data"
__C.model.kws_net_output_name = "prob"
__C.model.kws_chw_params = "1,64,192"
__C.model.kws_transpose = True

# asr
__C.model.asr_model_path = "/mnt/huanyuan/model/audio_model/hisi_model/asr_mandarin_taxi_16k/asr_mandarin_taxi_16k_64.caffemodel"
__C.model.asr_prototxt_path = "/mnt/huanyuan/model/audio_model/hisi_model/asr_mandarin_taxi_16k/asr_mandarin_taxi_16k_64_296.prototxt"
__C.model.asr_net_input_name = "data"
__C.model.asr_net_output_name = "prob"
__C.model.asr_chw_params = "1,396,64"
__C.model.asr_bpe = "/mnt/huanyuan/model/audio_model/hisi_model/asr_mandarin_taxi_16k/asr_mandarin_dict_taxi.txt"
__C.model.lm_path = "/mnt/huanyuan/model/audio_model/hisi_model/asr_mandarin_taxi_16k/3gram_asr_mandarin_taxi_408.bin" 


##################################
# test parameters
##################################

# 用于 RMAI_KWS_ASR_offline_API.py
__C.test = {}

# test_mode
# 0: input_wav
# 1: input_folder
__C.test.test_mode = 0

# input_Wav
# __C.test.input_wav = "/home/huanyuan/share/audio_data/mandarin_wav/taxi_phone/test/RM_Room_Taxi_Phone_S001T1P69.wav"
# __C.test.input_wav = "/mnt/huanyuan/data/speech/Recording/Daily_Record/mobile_phone/test/test-kws-xiaorui-asr-mandarin-taxi.wav"
__C.test.input_wav = "/home/huanyuan/share/audio_data/mandarin_wav/taxi_phone/no_response/taxi_no_response.wav"

# input_folder
__C.test.input_folder = ""

__C.test.output_folder = "/mnt/huanyuan/data/speech/Recording/demo_kws_asr_online_api/{}".format('-'.join('-'.join(str(datetime.now()).split('.')[0].split(' ')).split(':')))