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

__C.general.sample_rate = 8000
__C.general.nfilt = 64                              # 计算特征中，Mel 滤波器个数
__C.general.feature_freq = 56                       # 计算特征维度
__C.general.feature_time = 96                       # 每次送入 1s 数据，对应的特征时间维度 96

# kws
# activate bwc
__C.general.kws_feature_time = 120                  # kws 网络特征时间维度
__C.general.kws_stride_feature_time = 10            # kws 每间隔 10 个 feature_time 进行一次检索, 对应滑窗 100 ms，共检测 10 次
__C.general.kws_detection_threshold = 0.5           # kws 检测阈值 0.5
__C.general.kws_detection_number_threshold = 0.5    # kws 计数阈值 0.5
__C.general.kws_suppression_counter = 4             # kws 激活后抑制时间 4s

# asr mandarin taxi
__C.general.language_id = 0			                # 0： chinese  1： english
__C.general.asr_feature_time = 296                  # asr 网络特征时间维度，与语音特征容器长度相同
__C.general.asr_suppression_counter = 2             # asr 激活后抑制时间，间隔 2s 执行一次 asr 检测

__C.general.asr_second_on = False                   # asr 使用 bpe 和 phoneme 两个 model（该方法，目前只在 RMAI_KWS_ASR_options_BWC_bpe_phoneme.py 中使用）
__C.general.decode_id = 1			                # 0： greedy   1： beamsearch
__C.general.match_id = 0                            # 0:  bpe      1:  phoneme_robust  2:  phoneme_strict  3:  phoneme_combine (ONLF FOR ENGLISH)

# container
__C.general.audio_container_ms = 100                # 语音数据容器中，装有音频数据 100 ms
__C.general.audio_container_time = 10               # 语音数据容器中，装有音频数据 100 ms，对应特征维度 10
__C.general.feature_container_time = 396            # 语音特征容器中，装有时间维度 396
__C.general.feature_remove_after_time = 6           # 为保证特征一致，拼接特征需要丢弃最后的时间维度 6
__C.general.feature_remove_before_time = 100        # 为保证特征一致，拼接特征需要丢弃之前的时间维度 100

# on-off
__C.general.bool_output_wave = True

# init 
__C.general.window_size_samples = int(__C.general.sample_rate * __C.general.window_size_ms / 1000)
__C.general.window_stride_samples = int(__C.general.sample_rate * __C.general.window_stride_ms / 1000)
__C.general.window_container_samples = int(__C.general.sample_rate * __C.general.audio_container_ms / 1000)
__C.general.total_time_samples = int(__C.general.sample_rate * __C.general.total_time_ms / 1000)


##################################
# model parameters
##################################

__C.model = {}
__C.model.bool_caffe = True
# __C.model.bool_caffe = False
# __C.model.bool_pytorch = True

# kws
## caffe
__C.model.kws_model_path = ""
__C.model.kws_prototxt_path = ""
__C.model.kws_net_input_name = ""
__C.model.kws_net_output_name = ""
__C.model.kws_chw_params = ""
__C.model.kws_transpose = True

# asr
__C.model.asr_model_path = "/mnt/huanyuan/model/audio_model/novt_model/asr_mandarin_taxi_8k/asr_mandarin_taxi_8k_56.caffemodel"
__C.model.asr_prototxt_path = "/mnt/huanyuan/model/audio_model/novt_model/asr_mandarin_taxi_8k/asr_mandarin_taxi_8k_56_396.prototxt"
__C.model.asr_dict_path = "/mnt/huanyuan/model/audio_model/novt_model/asr_mandarin_taxi_8k/asr_mandarin_dict_taxi.txt"
__C.model.asr_lm_path = "/mnt/huanyuan/model/audio_model/novt_model/asr_mandarin_taxi_8k/3gram_asr_mandarin_taxi_408.bin"
__C.model.asr_net_input_name = "data"
__C.model.asr_net_output_name = "prob"
__C.model.asr_chw_params = "1,396,56"


##################################
# test parameters
##################################

# 用于 RMAI_KWS_ASR_offline_API.py
__C.test = {}

__C.test.input_wav = "/home/huanyuan/share/audio_data/cq_wav/taxi/cq1.wav"

__C.test.output_folder = "/mnt/huanyuan/data/speech/Recording/demo_kws_asr_online_api/{}".format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))