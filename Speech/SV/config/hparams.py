int16_max = (2 ** 15) - 1

TRAINING_NAME = "training"
TESTING_NAME = "testing"
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'

## Voice Activation Detection
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 30                  # In milliseconds，30 ms 音频用于 vad 计算
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out. 
vad_moving_average_width = 8            # 平滑长度，连续 8 帧平滑
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6              # 吕勇膨胀腐蚀思想，减少空洞现象

## Audio volume normalization
audio_norm_target_dBFS = -30            # 目标 dbFS，-30

# VoxCeleb1 nationalites
Anglophone_Nationalites = ["australia", "canada", "ireland", "uk", "usa"]