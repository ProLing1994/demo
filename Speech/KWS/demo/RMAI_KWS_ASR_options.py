from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

__C.general.window_size_ms = 1000                   # 每次送入 1s 数据
__C.general.window_stride_ms = 1000                 # 每次间隔 1s 时间
__C.general.total_time_ms = 3000                    # 算法处理时长 3s 时间

# kws
# activate bwc
__C.general.sample_rate = 16000
__C.general.feature_freq = 64                       # 计算特征维度
__C.general.nfilt = 64                              # 计算特征中，Mel 滤波器个数
__C.general.kws_feature_time = 196                  # kws 网络特征时间维度
__C.general.kws_stride_feature_time = 10            # kws 每间隔 10 个 feature_time 进行一次检索, 对应滑窗 100 ms
__C.general.kws_detection_threshold = 0.5           # kws 检测阈值 0.5
__C.general.kws_detection_number_threshold = 0.5    # kws 计数阈值 0.5
# __C.general.kws_detection_number_threshold = 0.3    # kws 计数阈值 0.3
# __C.general.kws_suppression_counter = 3             # kws 激活后抑制时间 3s
__C.general.kws_suppression_counter = 2             # kws 激活后抑制时间 2s
# __C.general.kws_suppression_counter = 1             # kws 激活后抑制时间 1s

# # # xiaoan8k/nihaoxiaoan8k
# __C.general.sample_rate = 8000
# __C.general.feature_freq = 48                       # 计算特征维度
# __C.general.nfilt = 48                              # 计算特征中，Mel 滤波器个数
# __C.general.kws_feature_time = 146                  # kws 网络特征时间维度
# __C.general.kws_stride_feature_time = 10            # kws 每间隔 10 个 feature_time 进行一次检索, 对应滑窗 100 ms
# __C.general.kws_detection_threshold = 0.5           # kws 检测阈值 0.5
# __C.general.kws_detection_number_threshold = 0.3    # kws 计数阈值 0.3
# __C.general.kws_suppression_counter = 3             # kws 激活后抑制时间 3s

# asr
__C.general.asr_feature_time = 296                  # asr 网络特征时间维度，与语音特征容器长度相同
__C.general.asr_suppression_counter = 2             # asr 激活后抑制时间，间隔 2s 执行一次 asr 检测

__C.general.audio_container_ms = 100                # 语音数据容器中，装有音频数据 100 ms
__C.general.feature_container_time = 296            # 语音特征容器中，装有时间维度 296
__C.general.feature_remove_after_time = 6           # 为保证特征一致，拼接特征需要丢弃最后的时间维度 6
__C.general.feature_remove_before_time = 100        # 为保证特征一致，拼接特征需要丢弃之前的时间维度 100

# on-off
# __C.general.bool_do_asr = True
__C.general.bool_do_asr = False
__C.general.bool_do_kws_weakup = True
__C.general.bool_output_wave = True
__C.general.bool_output_csv = False

# init 
__C.general.window_size_samples = int(__C.general.sample_rate * __C.general.window_size_ms / 1000)
__C.general.window_stride_samples = int(__C.general.sample_rate * __C.general.window_stride_ms / 1000)
__C.general.window_container_samples = int(__C.general.sample_rate * __C.general.audio_container_ms / 1000)
__C.general.total_time_samples = int(__C.general.sample_rate * __C.general.total_time_ms / 1000)