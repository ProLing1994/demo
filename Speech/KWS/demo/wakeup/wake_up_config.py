from easydict import EasyDict as edict
import numpy as np


__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

# model path
# xiaoyu
# __C.general.model_path = "./wakeup/model/kws_xiaoyu6_1_timeshift_spec_on_res15_11192020/"
__C.general.model_path = "./wakeup/model/kws_xiaoyu6_2_timeshift_spec_on_res15_11192020/"

# model epoch
__C.general.model_epoch = -1

# the number of GPUs used in training
__C.general.num_gpus = 1

# the GPUs' id 
__C.general.gpu_ids = '0'


##################################
# dataset parameters
##################################

__C.dataset = {}

# the number of input channel, currently only support 1 channel input
__C.dataset.input_channel = 1

# Number of audio samples per second
__C.dataset.sample_rate = 16000

# Length of each audio clip to be analyzed
__C.dataset.clip_duration_ms = 3000

# Duration of frequency analysis window
__C.dataset.window_size_ms = 30.0

# How far to move in time between frequency windows
__C.dataset.window_stride_ms = 10.0

# How the spectrogram is processed to produce features, support ["mfcc", "pcen", "fbank"]
__C.dataset.preprocess = "fbank"
# __C.dataset.preprocess = "mfcc"

# How many bins to use for the MFCC fingerprint
__C.dataset.feature_bin_count = 40

# input size of training data (w, h), unit: voxel
__C.dataset.data_size = [40, 301]


##################################
# label parameters
##################################

__C.dataset.label = {}

# label
__C.dataset.label.positive_label = ["xiaoyu"]
__C.dataset.label.negative_label = ["_silence_", "_unknown_"]
__C.dataset.label.negative_label_silence = __C.dataset.label.negative_label[0]
__C.dataset.label.negative_label_unknown = __C.dataset.label.negative_label[1]
__C.dataset.label.label_list = __C.dataset.label.negative_label + __C.dataset.label.positive_label
__C.dataset.label.num_classes = len(__C.dataset.label.positive_label) + len(__C.dataset.label.negative_label)


#####################################
# net
#####################################

__C.net = {}

__C.net.name = 'res15'


##################################
# test parameters
##################################

__C.test = {}

# mode
__C.test.method_mode = 1             # [0: RecognizeCommands, 1: RecognizeCommandsCountNumber]

__C.test.detection_threshold = 0.9          # [0.3,0.4,0.6,0.8,0.9,0.95]
__C.test.detection_number_threshold = 0.9   # [0.5,0.75,0.9]

__C.test.timeshift_ms = 30
__C.test.average_window_duration_ms = 800   # [450,800,1500]
__C.test.minimum_count = 15