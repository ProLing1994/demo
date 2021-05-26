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
__C.general.total_time_ms = 3000                    # 算法处理时长 3s 时间

__C.general.sample_rate = 8000
__C.general.nfilt = 48                              # 计算特征中，Mel 滤波器个数
__C.general.feature_freq = 48                       # 计算特征维度
__C.general.feature_time = 96                       # 每次送入 1s 数据，对应的特征时间维度 96

# kws
# xiaoan8k
__C.general.kws_feature_time = 146                  # kws 网络特征时间维度
__C.general.kws_stride_feature_time = 10            # kws 每间隔 10 个 feature_time 进行一次检索, 对应滑窗 100 ms，共检测 10 次
__C.general.kws_detection_threshold = 0.9           # kws 检测阈值 0.9
# __C.general.kws_detection_threshold = 0.5           # kws 检测阈值 0.5
__C.general.kws_detection_number_threshold = 0.5    # kws 计数阈值 0.5
__C.general.kws_suppression_counter = 1             # kws 激活后抑制时间 1s

# asr
__C.general.asr_feature_time = 296                  # asr 网络特征时间维度，与语音特征容器长度相同
__C.general.asr_suppression_counter = 2             # asr 激活后抑制时间，间隔 2s 执行一次 asr 检测

# container
__C.general.audio_container_ms = 100                # 语音数据容器中，装有音频数据 100 ms
__C.general.audio_container_time = 10               # 语音数据容器中，装有音频数据 100 ms，对应特征维度 10
__C.general.feature_container_time = 296            # 语音特征容器中，装有时间维度 296
__C.general.feature_remove_after_time = 6           # 为保证特征一致，拼接特征需要丢弃最后的时间维度 6
__C.general.feature_remove_before_time = 100        # 为保证特征一致，拼接特征需要丢弃之前的时间维度 100

# on-off
__C.general.bool_do_kws_weakup = True
__C.general.bool_do_asr = False
__C.general.bool_output_wave = True
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
# xiaoan8k
# __C.model.kws_model_path = "/mnt/huanyuan/model/audio_model/caffe_model/kws_xiaoan8k_res15/res15_1_5_05142021.caffemodel"
# __C.model.kws_prototxt_path = "/mnt/huanyuan/model/audio_model/caffe_model/kws_xiaoan8k_res15/res15_1_5_05142021.prototxt"
# __C.model.kws_label = "xiaoan8k"
# __C.model.kws_net_input_name = "blob1"
# __C.model.kws_net_output_name = "Softmax"
# __C.model.kws_chw_params = "1,146,48"
# __C.model.kws_transpose = False
__C.model.kws_model_path = "/mnt/huanyuan/model/audio_model/hisi_model/kws_xiaoan8k_tc_resnet14/kws_xiaoan8k_tc_resnet14_2_5_05202021.caffemodel"
__C.model.kws_prototxt_path = "/mnt/huanyuan/model/audio_model/hisi_model/kws_xiaoan8k_tc_resnet14/kws_xiaoan8k_tc_resnet14_2_5_05202021.prototxt"
# __C.model.kws_model_path = "/mnt/huanyuan/model/audio_model/novt_model/kws_xiaoan8k_tc_resnet14/kws_xiaoan8k_tc_resnet14_2_5_05202021.caffemodel"
# __C.model.kws_prototxt_path = "/mnt/huanyuan/model/audio_model/novt_model/kws_xiaoan8k_tc_resnet14/kws_xiaoan8k_tc_resnet14_2_5_05202021.prototxt"
__C.model.kws_label = "xiaoan8k"
__C.model.kws_net_input_name = "data"
__C.model.kws_net_output_name = "Softmax"
__C.model.kws_chw_params = "1,48,146"
__C.model.kws_transpose = True

# asr
__C.model.asr_model_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/english_0202_better.caffemodel"
__C.model.asr_prototxt_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/english_0202_mark.prototxt"
__C.model.asr_net_input_name = "data"
__C.model.asr_net_output_name = "conv39"
__C.model.asr_chw_params = "1,296,64"
__C.model.asr_bpe = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/english_bpe.txt"


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
# __C.test.input_wav = "/mnt/huanyuan/model/test_straming_wav/xiaoan8k_1_1_04082021_training_60.wav"
# __C.test.input_wav = "/mnt/huanyuan/model/test_straming_wav/xiaoan8k_1_1_04082021_validation_60.wav"
# __C.test.input_wav = "/mnt/huanyuan/model/test_straming_wav/xiaoan8k_1_3_04152021_validation.wav"
__C.test.input_wav = "/home/huanyuan/share/audio_data/weakup_xiaoan8k/test/xiaoan8k_1_1_04082021_validation_60.wav"

# input_folder
# __C.test.input_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/货车怠速场景/处理音频/"
# __C.test.input_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/其他录音/adpro/"
# __C.test.input_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/其他录音/adpro-1/"
__C.test.input_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/其他录音/adpro-2/"

# output_folder
# __C.test.output_folder = "/mnt/huanyuan/data/speech/Recording/demo_kws_asr_online_api/{}".format('-'.join('-'.join(str(datetime.now()).split('.')[0].split(' ')).split(':')))
# __C.test.output_folder = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoan8k_2_5_tc-resnet14-amba_fbankcpu_kd_05152021/test_straming_wav/实车录制_0427_caffemodel/阈值_05_05_唤醒间隔时间1s/货车怠速场景/"
# __C.test.output_folder = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoan8k_2_5_tc-resnet14-amba_fbankcpu_kd_05152021/test_straming_wav/实车录制_0427_caffemodel/阈值_09_05_唤醒间隔时间1s/货车怠速场景/"
# __C.test.output_folder = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoan8k_2_5_tc-resnet14-amba_fbankcpu_kd_05152021/test_straming_wav/实车录制_0427_caffemodel/阈值_05_05_唤醒间隔时间1s/其他录音/adpro/"
# __C.test.output_folder = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoan8k_2_5_tc-resnet14-amba_fbankcpu_kd_05152021/test_straming_wav/实车录制_0427_caffemodel/阈值_05_05_唤醒间隔时间1s/其他录音/adpro-1/"
__C.test.output_folder = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoan8k_2_5_tc-resnet14-amba_fbankcpu_kd_05152021/test_straming_wav/实车录制_0427_caffemodel/阈值_05_05_唤醒间隔时间1s/其他录音/adpro-2/"