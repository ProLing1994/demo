from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

# data folder
__C.general.data_dir = "/mnt/huanyuan/data/speech/kws/tf_speech_commands/speech_commands/"
__C.general.sub_data_dir = []

# data version
__C.general.version = "1.0"

# data date
__C.general.date = "02042021"

# data path
__C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/tf_speech_commands/dataset_1.0_02042021/total_data_files.csv"

# background noise path
__C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/tf_speech_commands/dataset_1.0_02042021/background_noise_files.csv"

# test after save pytorch model
__C.general.is_test = True

# the output of training models and logging files
__C.general.save_dir = "/home/huanyuan/model/kws_test"
# __C.general.save_dir = "/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_10212020_le-4/"
# __C.general.save_dir = "/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_cnn-tpool2_10222020/"
# __C.general.save_dir = "/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_cnn-one-fstride1_10222020/"
# __C.general.save_dir = "/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res15_10232020/"
# __C.general.save_dir = "/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res15-narrow_10232020/"
# __C.general.save_dir = "/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res8_10232020/"
# __C.general.save_dir = "/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res8-narrow_10232020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_speech_commands_res15_02042021/"

# finetune model
__C.general.finetune_on = False
__C.general.finetune_model_dir = ""
__C.general.finetune_epoch = 0

# set certain epoch to continue training, set -1 to train from scratch
__C.general.resume_epoch = -1

# the number of GPUs used in training
__C.general.num_gpus = 1

# the GPUs' id used in training
# __C.general.gpu_ids = '7'
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
__C.knowledge_distillation.teacher_model_dir = ""
__C.knowledge_distillation.epoch = 0

# loss
# kd: https://github.com/peterliht/knowledge-distillation-pytorch
# cd: https://github.com/zhouzaida/channel-distillation
__C.knowledge_distillation.loss_name = 'kd'

# kd, alpha
__C.knowledge_distillation.alpha = 0.95

# kd, temperature
__C.knowledge_distillation.temperature = 6


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of input channel, currently only support 1 channel input
__C.dataset.input_channel = 1

# Number of audio samples per second
__C.dataset.sample_rate = 16000

# Length of each audio clip to be analyzed
__C.dataset.clip_duration_ms = 1000

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
__C.dataset.data_size = [40, 101]


##################################
# label parameters
##################################

__C.dataset.label = {}

# label
__C.dataset.label.positive_label = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
__C.dataset.label.positive_label_chinese_name_list = [""]
__C.dataset.label.positive_label_together = False
__C.dataset.label.positive_label_together_label = ["positive"]
__C.dataset.label.negative_label = ["_silence_", "_unknown_"]
__C.dataset.label.negative_label_silence = __C.dataset.label.negative_label[0]
__C.dataset.label.negative_label_unknown = __C.dataset.label.negative_label[1]
__C.dataset.label.label_list = __C.dataset.label.negative_label + __C.dataset.label.positive_label
__C.dataset.label.num_classes = len(__C.dataset.label.positive_label) + len(__C.dataset.label.negative_label)

# label percentage
__C.dataset.label.silence_percentage = 10.0     # 10%
__C.dataset.label.unknown_percentage = 10.0     # 10%
# __C.dataset.label.difficult_sample_mining = True
__C.dataset.label.difficult_sample_mining = False
__C.dataset.label.difficult_sample_percentage = 10.0

# trian/validation/test percentage
__C.dataset.label.validation_percentage = 10.0  # 10%
__C.dataset.label.testing_percentage = 10.0     # 10%


##################################
# data augmentation parameters
##################################

# data augmentation parameters
__C.dataset.augmentation = {}

# on
__C.dataset.augmentation.on = True
# __C.dataset.augmentation.on = False

# How many of the training samples have background noise mixed in.
__C.dataset.augmentation.background_frequency = 0.8

# How loud the background noise should be, between 0 and 1.
__C.dataset.augmentation.background_volume = 0.1

# Range to randomly shift the training audio by in time(ms).
__C.dataset.augmentation.time_shift_ms = 100.0

# Time shift enhancement multiple of negative samples, which is effective for advanced prediction and lag prediction
__C.dataset.augmentation.time_shift_multiple = 1

# based on audio waveform: on, just for positive samples.
# __C.dataset.augmentation.speed_volume_on = True
__C.dataset.augmentation.speed_volume_on = False

# How fast the audio should be, just for positive samples.
__C.dataset.augmentation.possitive_speed = '0.9,1.0,1.1'

# How loud the audio should be, just for positive samples.
__C.dataset.augmentation.possitive_volume = '0.4,0.7,1.0,1.3,1.6'

# based on audio spectrum: on
# __C.dataset.augmentation.spec_on = True
__C.dataset.augmentation.spec_on = False
__C.dataset.augmentation.F = 5
__C.dataset.augmentation.T = 20
__C.dataset.augmentation.num_masks = 1

####################################
# training loss
####################################

__C.loss = {}

# the loss name, support ['softmax','focal']
__C.loss.name = 'softmax'
# __C.loss.name = 'focal'

# the weight matrix for each class in focal loss, including background class
__C.loss.obj_weight = np.array([[1/9, 0, 0], [0, 1/9, 0], [0, 0, 7/9]])

# the gamma parameter in focal loss
__C.loss.focal_gamma = 2


#####################################
# net
#####################################

__C.net = {}

# the network name
__C.net.class_name = "SpeechResModel"
# __C.net.model_name = 'cnn-trad-pool2'
# __C.net.model_name = 'cnn-one-fstride1'
# __C.net.model_name = 'cnn-tpool2'
# __C.net.model_name = 'res15'
# __C.net.model_name = 'res15-narrow'
# __C.net.model_name = 'res8'
# __C.net.model_name = 'res8-narrow'
# __C.net.model_name = 'lstm-avg'
# __C.net.model_name = 'lstm-attention'
# __C.net.model_name = 'crnn-avg'
# __C.net.model_name = 'crnn-attention'
# __C.net.model_name = 'edge-speech-nets'
# __C.net.model_name = 'tc-resnet8'
__C.net.model_name = 'tc-resnet14'


######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
# __C.train.num_epochs = 4000
# __C.train.num_epochs = 2000
# __C.train.num_epochs = 500
__C.train.num_epochs = 1

# the number of samples in a batch
# __C.train.batch_size = 2048
# __C.train.batch_size = 256
# __C.train.batch_size = 16
__C.train.batch_size = 1

# the number of threads for IO
# __C.train.num_threads = 64
# __C.train.num_threads = 16
__C.train.num_threads = 1

# the number of batches to update loss curve
__C.train.plot_snapshot = 5

# the number of epochs to save model
__C.train.save_epochs = 25
# __C.train.save_epochs = 1


######################################
# learning rate parameters
######################################

# learning rate = lr*gamma**(epoch//step_size)
# __C.train.lr = 1e-3
__C.train.lr = 1e-4

# step size for step learning rate
__C.train.lr_step_size = 0

# gamma for learning rate
__C.train.lr_gamma = 0.9


######################################
# optimizer parameters
######################################

# optimizer, support SGD and Adam
__C.train.optimizer = 'Adam'

# SGD, Adam momentum
__C.train.momentum = 0.9

# SGD,Adam weight decay
# __C.train.weight_decay = 0.0
__C.train.weight_decay = 0.0001

# the beta in Adam optimizer
__C.train.betas = (0.9, 0.999)


######################################
# debug parameters
######################################

__C.debug = {}

# whether to save input images
# __C.debug.save_inputs = True
__C.debug.save_inputs = False

# the number of processing for save input images
# __C.debug.num_processing = 64
__C.debug.num_processing = 16

# random seed used in training
__C.debug.seed = 0