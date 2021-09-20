from easydict import EasyDict as edict
import numpy as np
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/TTS')
from dataset.symbols import *

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

__C.general.dataset_list = ['librispeech_clean_360', 'librispeech_clean_100', 'test_clean']
# __C.general.dataset_list = ['test']
__C.general.dataset_path_dict = {"librispeech_clean_360_training": "/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech/train-clean-360",
                                "librispeech_clean_100_training": "/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech/train-clean-100",
                                "test_clean_testing": "/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech/test-clean",
                                }

# data path
__C.general.data_dir = "/mnt/huanyuan/data/speech/tts/dataset/"

# the output of training models and logging files
__C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/tts/test/"

# test after save pytorch model
__C.general.is_test = True

# finetune model
__C.general.finetune_on = False
__C.general.finetune_model_dir = ""
__C.general.finetune_epoch = 0

# set certain epoch to continue training, set -1 to train from scratch
__C.general.resume_epoch = -1

# the number of GPUs used in training
__C.general.num_gpus = 1

# the GPUs' id used in training
__C.general.gpu_ids = '0'

# data_parallel_mode: [0, 1]
# 0: 单机多卡，DataParallel
# 1: 单/多级多卡、分布式，DistributedDataParallel
__C.general.data_parallel_mode = 0


##################################
# speaker verification parameters
##################################

__C.speaker_verification = {}

__C.speaker_verification.model_name = 'basic'
__C.speaker_verification.class_name = 'SpeakerEncoder'
__C.speaker_verification.model_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/sv/ti_sv_1_0_09142021/"
__C.speaker_verification.model_path = "SV."
__C.speaker_verification.epoch = 250


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of input channel, currently only support 1 channel input
__C.dataset.input_channel = 1

# Number of audio samples per second
__C.dataset.sample_rate = 16000

# Length of each audio clip to be analyzed
__C.dataset.clip_duration_ms = 1600

# Duration of frequency analysis window
__C.dataset.window_size_ms = 32.0

# How far to move in time between frequency windows
__C.dataset.window_stride_ms = 10.0

# How the spectrogram is processed to produce features, support ["mfcc", "pcen", "fbank", "fbank"]
__C.dataset.preprocess = "fbank_cpu"

# How many bins to use for the Mel feature
__C.dataset.feature_bin_count = 64

# How many nfilt to use for the Mel feature, only support preprocess=fbank_cpu
__C.dataset.nfilt = 64

# input size of training data (w, h), whether input size is a multiple of 16, unit: voxel
# __C.dataset.h_alignment = True, [hisi], 模型需要图像输入长度为 16 的倍数
# __C.dataset.h_alignment = False, [amba, novt]
__C.dataset.w_alignment = False
__C.dataset.h_alignment = False

# input size of training data (w, h), unit: voxel
__C.dataset.data_size = [64, 156]

# num_chars
__C.dataset.num_chars = len(symbols)

# speaker_embedding_size
__C.dataset.speaker_embedding_size = 256

# tts_cleaner_names
__C.dataset.tts_cleaner_names = ["english_cleaners"]


##################################
# data augmentation parameters
##################################

# data augmentation parameters
__C.dataset.augmentation = {}

# based on audio waveform: on
# __C.dataset.augmentation.on = True
__C.dataset.augmentation.on = False

# How many of the training samples have background noise mixed in.
__C.dataset.augmentation.background_frequency = 0.8

# How loud the background noise should be, between 0 and 1.
__C.dataset.augmentation.background_volume = 0.1

# How many of the training samples have synthetic noise mixed in.
__C.dataset.augmentation.synthetic_frequency = 0.8

# type of the synthetic noise, support ['white', 'salt_pepper'].
# __C.dataset.augmentation.synthetic_type = 'white'
__C.dataset.augmentation.synthetic_type = 'salt_pepper'

# the scale parameter in white synthetic noise
__C.dataset.augmentation.synthetic_scale = 0.001

# the prob parameter in salt pepper synthetic noise
__C.dataset.augmentation.synthetic_prob = 0.001

# Range to randomly shift the training audio by in time(ms).
__C.dataset.augmentation.time_shift_ms = 100.0

# based on audio waveform: on.
# __C.dataset.augmentation.speed_volume_on = True
__C.dataset.augmentation.speed_volume_on = False

# How fast the audio should be.
__C.dataset.augmentation.speed = [0.9, 1.1]

# How loud the audio should be.
__C.dataset.augmentation.volume = [1.0, 1.0]

# How pitch the audio should be.
# __C.dataset.augmentation.pitch_on = True
__C.dataset.augmentation.pitch_on = False
__C.dataset.augmentation.pitch = [-5, 5]

# based on audio vtlp: on
# vtlp: http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=34DDD4B0CDCE76942A879204E8B7716C?doi=10.1.1.369.733&rep=rep1&type=pdf
# __C.dataset.augmentation.vtlp_on = True
__C.dataset.augmentation.vtlp_on = False

# based on audio spectrum: on
# spec_aug
# __C.dataset.augmentation.spec_on = True
__C.dataset.augmentation.spec_on = False
__C.dataset.augmentation.F = 5
__C.dataset.augmentation.T = 10
__C.dataset.augmentation.num_masks = 2


#####################################
# net
#####################################

__C.net = {}

# the network name
__C.net.model_name = 'tacotron'
__C.net.class_name = "Tacotron"

# r frames
__C.net.r = 2

######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
__C.train.num_epochs = 10

# the number of samples in a batch
__C.train.batch_size = 1

# the number of threads for IO
__C.train.num_threads = 1

# the number of batches to show log
__C.train.show_log = 5

# the number of batches to update loss curve
__C.train.plot_snapshot = 5

# the number of epochs to save model
__C.train.save_epochs = 1


######################################
# learning rate parameters
######################################

# learning rate = lr*gamma**(epoch//step_size)
__C.train.lr = 1e-4

# step size for step learning rate
__C.train.lr_step_size = 0

# gamma for learning rate
__C.train.lr_gamma = 0.9


######################################
# optimizer parameters
######################################

# optimizer, support SGD and Adam
# __C.train.optimizer = 'SGD'
__C.train.optimizer = 'Adam'

# SGD momentum
__C.train.momentum = 0.9

# SGD, Adam weight decay
# __C.train.weight_decay = 0.0
__C.train.weight_decay = 0.0001

# the beta in Adam optimizer
__C.train.betas = (0.9, 0.999)


####################################
# training loss
####################################

__C.loss = {}

__C.loss.name = 'None'

# EMA: expontential moving average on
# EMA: https://github.com/ProLing1994/pytorch-loss/blob/master/ema.py
# __C.loss.ema_on = True
__C.loss.ema_on = False

# the alpha parameter in EMA: each parameter p should be computed as p_hat = alpha * p + (1. - alpha) * p_hat
__C.loss.ema_alpha = 0.995


##################################
# regularization parameters
##################################

# regularization parameters
__C.regularization = {}

# regularization: label smoothing parameters
__C.regularization.label_smoothing = {}

# regularization: label smoothing on
__C.regularization.label_smoothing.on = False

# regularization: label smoothing epsilon 
__C.regularization.label_smoothing.epsilon = 0.1


######################################
# debug parameters
######################################

__C.debug = {}

# random seed used in training
__C.debug.seed = 0