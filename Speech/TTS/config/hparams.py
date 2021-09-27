int16_max = (2 ** 15) - 1

TRAINING_NAME = "training"
TESTING_NAME = "testing"

## Voice Activation Detection
trim_silence = True
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 30                  # In milliseconds，30 ms 音频用于 vad 计算
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out. 
vad_moving_average_width = 8            # 平滑长度，连续 8 帧平滑
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6              # 利用膨胀腐蚀思想，减少空洞现象

## Audio volume normalization
normalize = False
audio_norm_target_dBFS = -30            # 目标 dbFS，-30

### Signal Processing (used in both synthesizer and vocoder)
min_level_db = -100
ref_level_db = 20
max_abs_value = 4.                         # Gradient explodes if too big, premature convergence if too small.
preemphasis = 0.97                         # Filter coefficient to use if preemphasize is True
preemphasize = True

### Data Preprocessing
rescale = True
rescaling_max = 0.9

clip_mels_length = True                    # If true, discards samples exceeding max_mel_frames
max_mel_frames = 900    

### Audio processing options
signal_normalization = True
allow_clipping_in_normalization = True     # Used when signal_normalization = True
symmetric_mels = True                      # Sets mel range to [-max_abs_value, max_abs_value] if True,
                                           #               and [0, max_abs_value] if False

### Griffin-Lim
power = 1.5
griffin_lim_iters = 60

### SV2TTS
silence_min_duration_split = 0.4           # Duration in seconds of a silence for an utterance to be split
utterance_min_duration = 1.6               # Duration in seconds below which utterances are discarded