from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

'''
SLR38: http://www.openslr.org/38/, 854/0(854)
SLR68: http://www.openslr.org/68/, 1016/43(1059)
Aishell3: http://www.openslr.org/93/, 171/156(218)
CN-Celeb1: https://www.openslr.org/82/, 992/0(992)
CN-Celeb2: https://www.openslr.org/82/, 1996/0(1996)
'''
# __C.general.dataset_list = ['SLR38', 'SLR68', 'Aishell3', 'CN-Celeb1', 'CN-Celeb2']
__C.general.dataset_list = ['Aishell3']
# __C.general.dataset_list = ['test']
__C.general.dataset_path_dict = {
                                    "SLR38": "/mnt/huanyuan/data/speech/asr/Chinese/SLR38/ST-CMDS-20170001_1-OS/",
                                    "SLR38_training": "/mnt/huanyuan/data/speech/asr/Chinese/SLR38/ST-CMDS-20170001_1-OS/",
                                    "SLR38_testing": None,
                                    "SLR68": "/mnt/huanyuan/data/speech/asr/Chinese/SLR68/",
                                    "SLR68_training": "/mnt/huanyuan/data/speech/asr/Chinese/SLR68/train",
                                    "SLR68_testing": "/mnt/huanyuan/data/speech/asr/Chinese/SLR68/test",
                                    "Aishell3": "/mnt/huanyuan/data/speech/asr/Chinese/Aishell3/", 
                                    "Aishell3_training": "/mnt/huanyuan/data/speech/asr/Chinese/Aishell3/train/wav", 
                                    "Aishell3_testing": "/mnt/huanyuan/data/speech/asr/Chinese/Aishell3/test/wav", 
                                    "CN-Celeb1": "/mnt/huanyuan/data/speech/sv/CN-Celeb1/CN-Celeb_flac/data/",
                                    "CN-Celeb1_training": "/mnt/huanyuan/data/speech/sv/CN-Celeb1/CN-Celeb_flac/data/",
                                    "CN-Celeb1_testing": None,
                                    "CN-Celeb2": "/mnt/huanyuan/data/speech/sv/CN-Celeb2/CN-Celeb2_flac/data/",
                                    "CN-Celeb2_training": "/mnt/huanyuan/data/speech/sv/CN-Celeb2/CN-Celeb2_flac/data/",
                                    "CN-Celeb2_testing": None,
                                    "background_noise":"/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/KwsEnglishDataset/_background_noise_",
                                    }

# data path
__C.general.data_dir = "/mnt/huanyuan2/data/speech/sv/Chinese_TI_SV_dataset/dataset/"

# the output of training models and logging files
# __C.general.save_dir = "/mnt/huanyuan2/model/sv/Chinese_TI_SV/test"
# __C.general.save_dir = "/mnt/huanyuan2/model/sv/Chinese_TI_SV/ti_sv_1_0_basic_10122021"
__C.general.save_dir = "/mnt/huanyuan2/model/sv/Chinese_TI_SV/ti_sv_1_1_basic_10122021"

# test after save pytorch model
__C.general.is_test = True
# __C.general.is_test = False

# finetune model
__C.general.finetune_on = True
# __C.general.finetune_on = False

# 模型加载方式，[0: 方式一, 1: 方式二]
__C.general.load_mode_type = 0

# 方式一：加载模型训练过程中保存模型
__C.general.finetune_model_dir = "/mnt/huanyuan2/model/sv/Chinese_TI_SV/ti_sv_1_0_basic_10122021/"
__C.general.finetune_epoch = 400
# 方式二：加载其他模型结构
__C.general.finetune_model_path = ""
__C.general.finetune_model_state = 'model_state'
__C.general.finetune_ignore_key_list = []

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
# knowledge distillation parameters
##################################

__C.knowledge_distillation = {}

# knowledge distillation: on
__C.knowledge_distillation.on = False

# teacher model
__C.knowledge_distillation.teacher_model_name = ''
__C.knowledge_distillation.teacher_class_name = ''
__C.knowledge_distillation.teacher_model_dir = ""
__C.knowledge_distillation.epoch = 0


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of input channel, currently only support 1 channel input
__C.dataset.input_channel = 1

# Number of audio samples per second
__C.dataset.sample_rate = 16000

# Length of each audio clip to be analyzed
__C.dataset.clip_duration_ms = 5000

# Dynamic length ratio of each audio clip to be analyzed
# 由于模型的特殊性，输入音频数据可以是变长的
__C.dataset.clip_duration_dynamic_length_on = True
__C.dataset.clip_duration_dynamic_ratio = [0.5, 1.0]

# Duration of frequency analysis window
__C.dataset.window_size_ms = 50.0

# How far to move in time between frequency windows
__C.dataset.window_stride_ms = 12.5

# How the spectrogram is processed to produce features, support ["fbank", "fbank_log", "fbank_log_manual", "pcen", "fbank_cpu"]
__C.dataset.preprocess = "fbank_log_manual"

# How many bins to use for the Mel feature
__C.dataset.feature_bin_count = 80

# How many nfilt to use for the Mel feature, only support preprocess=fbank_cpu
__C.dataset.nfilt = 80

# fmin, only support preprocess ["fbank_log", "fbank_log_manual"]
# Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To 
# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
__C.dataset.fmin = 55

# fmax, only support preprocess ["fbank_log", "fbank_log_manual"]
# To be increased/reduced depending on data.
__C.dataset.fmax = 7600

# input size of training data (w, h), whether input size is a multiple of 16, unit: voxel
# __C.dataset.h_alignment = True, [hisi], 模型需要图像输入长度为 16 的倍数
# __C.dataset.h_alignment = False, [amba, novt]
__C.dataset.w_alignment = False
__C.dataset.h_alignment = False

# input size of training data (w, h), unit: voxel
__C.dataset.data_size = [80, 401]


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
__C.net.model_name = "/home/huanyuan/code/demo/Speech/SV/network/basic.py"
# __C.net.model_name = "/home/huanyuan/code/demo/Speech/SV/network/res34.py"
__C.net.class_name = 'SpeakerEncoder'


######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
__C.train.num_epochs = 1000

# the number of samples in a batch
__C.train.speakers_per_batch = 64
# __C.train.speakers_per_batch = 4
__C.train.utterances_per_speaker = 10
__C.train.batch_size = __C.train.speakers_per_batch

# the number of threads for IO
# __C.train.num_threads = 1
__C.train.num_threads = 2

# the number of batches to show log
__C.train.show_log = 5

# the number of batches to update loss curve
__C.train.plot_snapshot = 5

# the number of epochs to plot umap
__C.train.plot_umap = 1

# the number of epochs to save model
__C.train.save_epochs = 5


######################################
# learning rate parameters
######################################

# learning rate = lr*gamma**(epoch//step_size)
__C.train.lr = 1e-3
# __C.train.lr = 1e-4

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

# the loss name, support ['softmax','focal']
__C.loss.name = 'softmax'
# __C.loss.name = 'focal'

# the number of class
__C.loss.num_classes =  __C.train.batch_size

# the weight matrix for each class in focal loss, including background class
__C.loss.obj_weight = None

# the gamma parameter in focal loss
__C.loss.focal_gamma = 2

# EMA: expontential moving average on
# EMA: https://github.com/ProLing1994/pytorch-loss/blob/master/ema.py
__C.loss.ema_on = True
# __C.loss.ema_on = False

# the alpha parameter in EMA: each parameter p should be computed as p_hat = alpha * p + (1. - alpha) * p_hat
__C.loss.ema_alpha = 0.995

# loss
# kd: https://github.com/peterliht/knowledge-distillation-pytorch
# cd: https://github.com/zhouzaida/channel-distillation
__C.knowledge_distillation.loss_name = 'kd'

# kd, alpha
__C.knowledge_distillation.alpha = 0.95

# kd, temperature
__C.knowledge_distillation.temperature = 6


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


##################################
# test parameters
##################################

__C.test = {}

# the number of testing epochs
__C.test.model_epoch = -1
# __C.test.model_epoch = 999
