from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

'''
librispeech_other: 1166/33
VoxCeleb1: 1088/35
VoxCeleb2: 5994/118
'''
# __C.general.TISV_dataset_list = ['librispeech_other', 'VoxCeleb1', 'VoxCeleb2']
__C.general.TISV_dataset_list = ['test']
__C.general.TISV_dataset_path_dict = {"librispeech_other_training": "/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech/train-other-500",
                                    "librispeech_other_testing": "/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech/test-other",
                                    "VoxCeleb1_training": "/mnt/huanyuan/data/speech/sv/VoxCeleb1/wav",
                                    "VoxCeleb1_testing": "/mnt/huanyuan/data/speech/sv/VoxCeleb1/test_wav",
                                    "VoxCeleb1_csv": "/mnt/huanyuan/data/speech/sv/VoxCeleb1/vox1_meta.csv",
                                    "VoxCeleb2_training": "/mnt/huanyuan/data/speech/sv/VoxCeleb2/dev/aac",
                                    "VoxCeleb2_testing": "/mnt/huanyuan/data/speech/sv/VoxCeleb2/test/aac",
                                    "VoxCeleb2_csv": "/mnt/huanyuan/data/speech/sv/VoxCeleb2/vox2_meta.csv",
                                    "background_noise":"/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/KwsEnglishDataset/_background_noise_",
                                    }

# data path
__C.general.data_dir = "/mnt/huanyuan/data/speech/sv/TI_SV_dataset/dataset/"

# the output of training models and logging files
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/sv/test_0912/"
__C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/sv/ti_sv_1_0_09142021/"

# test after save pytorch model
__C.general.is_test = True

# finetune model
# 方式一：模型训练过程中，保存模型
__C.general.finetune_on = True
__C.general.finetune_model_dir = ""
__C.general.finetune_epoch = 0
# 方式二：加载其他模型结构
__C.general.finetune_model_path = "/mnt/huanyuan/model/model_10_30_25_21/model/sv/pretrained/pretrain_model/parameter.pkl"
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
__C.dataset.clip_duration_ms = 1600

# Duration of frequency analysis window
# __C.dataset.window_size_ms = 32.0
__C.dataset.window_size_ms = 25.0

# How far to move in time between frequency windows
__C.dataset.window_stride_ms = 10.0

# How the spectrogram is processed to produce features, support ["fbank", "fbank_log", "pcen", "fbank_cpu"]
# __C.dataset.preprocess = "fbank_cpu"
__C.dataset.preprocess = "fbank"

# How many bins to use for the Mel feature
# __C.dataset.feature_bin_count = 64
__C.dataset.feature_bin_count = 40

# How many nfilt to use for the Mel feature, only support preprocess=fbank_cpu
# __C.dataset.nfilt = 64
__C.dataset.nfilt = 40

# input size of training data (w, h), whether input size is a multiple of 16, unit: voxel
# __C.dataset.h_alignment = True, [hisi], 模型需要图像输入长度为 16 的倍数
# __C.dataset.h_alignment = False, [amba, novt]
__C.dataset.w_alignment = False
__C.dataset.h_alignment = False

# input size of training data (w, h), unit: voxel
# __C.dataset.data_size = [64, 156]
__C.dataset.data_size = [40, 160]


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
__C.net.model_name = 'basic'
__C.net.class_name = 'SpeakerEncoder'


######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
__C.train.num_epochs = 100

# the number of samples in a batch
__C.train.speakers_per_batch = 4
__C.train.utterances_per_speaker = 10
__C.train.batch_size = __C.train.speakers_per_batch

# the number of threads for IO
__C.train.num_threads = 1

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

# the loss name, support ['softmax','focal']
# __C.loss.name = 'softmax'
__C.loss.name = 'focal'

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
