int16_max = (2 ** 15) - 1

TRAINING_NAME = "training"
TESTING_NAME = "testing"
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'

### 数据预处理阶段
## Audio volume normalization
# 在数据预处理阶段，将音频音量大小归一化
normalize = True
audio_norm_target_dBFS = -30                # 目标 dbFS，-30

## Voice Activation Detection
# 在数据预处理阶段，Vad 消除静音音频
trim_silence = True
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 30                      # In milliseconds，30 ms 音频用于 vad 计算
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out. 
vad_moving_average_width = 8                # 平滑长度，连续 8 帧平滑
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6                  # 利用膨胀腐蚀思想，减少空洞现象

## Check wave length
# 在数据预处理阶段，需要检查数据的长度，若数据长度太短，则删除数据
check_wave_length_ms = 1600

## VoxCeleb1 nationalites
# 在数据预处理阶段，对 VoxCeleb1 数据集进行预处理
Anglophone_Nationalites = ["australia", "canada", "ireland", "uk", "usa"]

### 模型训练阶段
## Trim silence
# 在模型训练阶段，利用函数 librosa.effects.trim 修剪静音片段
# M-AILABS (and other datasets) trim params (these parameters are usually correct for any 
# data, but definitely must be tuned for specific speakers)
trim_fft_size=512
trim_hop_size=128
trim_top_db=23

## Data rescale
# 在模型训练阶段，音频音量大小缩放
rescale = True                              # Whether to rescale audio prior to preprocessing
rescaling_max = 0.9                         # Rescaling value

## Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
# train samples of lengths between 3sec and 14sec are more than enough to make a model capable
# of good parallelization.
clip_mels_length = True                     # If true, discards samples exceeding max_mel_frames
max_mel_frames = 900    

## Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude 
# levels. Also allows for better G&L phase reconstruction)
# 在模型训练阶段，特征计算参数，预加重
preemphasize = True                         # whether to apply filter
preemphasis = 0.97                          # Filter coefficient to use if preemphasize is True

## Limits
# 在模型训练阶段，特征计算参数
min_level_db = -100
ref_level_db = 20

## Mel and Linear spectrograms normalization/scaling and clipping
# 在模型训练阶段，特征计算参数，对 Mel 频率值归一化
signal_normalization = True
# Whether to normalize mel spectrograms to some predefined range (following below parameters)
allow_clipping_in_normalization = True      # Only relevant if signal_normalization = True
# Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, 
# faster and cleaner convergence)
symmetric_mels = True                       # Sets mel range to [-max_abs_value, max_abs_value] if True,
                                            #               and [0, max_abs_value] if False
# max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not 
# be too big to avoid gradient explosion, not too small for fast convergence)
max_abs_value = 4.                          # Gradient explodes if too big, premature convergence if too small.

## Griffin-Lim
# Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
power = 1.5
# Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
griffin_lim_iters = 60