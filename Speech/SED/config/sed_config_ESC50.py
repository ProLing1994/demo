from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

__C.general.data_dir = "/mnt/huanyuan/data/speech/sed/sound_classification_50/original_dataset/"
__C.general.sub_data_dir = []

# data version
__C.general.version = "1.0"

# data date
__C.general.date = "04152021"

# data path
__C.general.data_csv_path = "/mnt/huanyuan/data/speech/sed/sound_classification_50/experimental_dataset/dataset_1.0_04152021/train_test_dataset.csv"

# background noise path
__C.general.background_data_path = ""

# test after save pytorch model
__C.general.is_test = True

# the output of training models and logging files
__C.general.save_dir = "/mnt/huanyuan/model/sed_test"

# finetune model
__C.general.finetune_on = False
__C.general.finetune_model_dir = ""
__C.general.finetune_epoch = 0

# set certain epoch to continue training, set -1 to train from scratch
__C.general.resume_epoch = -1

# the number of GPUs used in training
# __C.general.num_gpus = 4
__C.general.num_gpus = 1

# the GPUs' id used in training
# __C.general.gpu_ids = '4, 5, 6, 7'
# __C.general.gpu_ids = '6, 7'
__C.general.gpu_ids = '0'

# data_parallel_mode: [0, 1]
# 0: 单机多卡，DataParallel
# 1: 单/多级多卡、分布式，DistributedDataParallel
__C.general.data_parallel_mode = 0

##################################
# data set parameters
##################################

__C.dataset = {}

# Number of audio samples per second
__C.dataset.sample_rate = 32000

# Length of each audio clip to be analyzed
__C.dataset.clip_duration_ms = 5000

# Duration of frequency analysis window
__C.dataset.window_size_ms = 32.0

# How far to move in time between frequency windows
__C.dataset.window_stride_ms = 10.0

# How the spectrogram is processed to produce features, support ["mfcc", "pcen", "fbank", "fbank"]
# __C.dataset.preprocess = "fbank"
# __C.dataset.preprocess = "pcen"
# __C.dataset.preprocess = "mfcc"
__C.dataset.preprocess = "fbank_cpu"

# How many bins to use for the Mel feature
__C.dataset.feature_bin_count = 64

# How many nfilt to use for the Mel feature, only support preprocess=fbank_cpu
__C.dataset.nfilt = 64

# the number of input channel, currently only support 1 channel input
__C.dataset.input_channel = 1

# input size of training data (w, h), unit: voxel
# __C.dataset.data_size = [64, 501]
__C.dataset.data_size = [64, 496]

##################################
# label parameters
##################################

__C.dataset.label = {}

# label
__C.dataset.label.num_classes = 50

# 多分类任务或者多标签任务, support ["multi_class", "multi_label"]
__C.dataset.label.type = 'multi_class'

##################################
# data augmentation parameters
##################################

# data augmentation parameters
__C.dataset.augmentation = {}

# based on audio waveform: on
# __C.dataset.augmentation.on = True
__C.dataset.augmentation.on = False

# based on audio waveform: on.
__C.dataset.augmentation.background_noise_on = False

# How many of the training samples have background noise mixed in.
__C.dataset.augmentation.background_frequency = -1

# How loud the background noise should be, between 0 and 1.
__C.dataset.augmentation.background_volume = -1

# based on audio waveform: on.
__C.dataset.augmentation.time_shift_on = False

# Range to randomly shift the training audio by in time(ms).
__C.dataset.augmentation.time_shift_ms = 1000.0

# based on audio waveform: on.
# __C.dataset.augmentation.speed_volume_on = True
__C.dataset.augmentation.speed_volume_on = False

# How fast the audio should be.
__C.dataset.augmentation.speed = [0.9, 1.1]

# How loud the audio should be.
__C.dataset.augmentation.volume = [0.4, 1.6]

# How pitch the audio should be.
# __C.dataset.augmentation.pitch_on = True
__C.dataset.augmentation.pitch_on = False
__C.dataset.augmentation.pitch = [-5, 5]

# based on audio spectrum: on
# __C.dataset.augmentation.spec_on = True
__C.dataset.augmentation.spec_on = False

# spectrum augmentation
__C.dataset.augmentation.F = 5
__C.dataset.augmentation.T = 20
__C.dataset.augmentation.num_masks = 1

# minx up
# __C.dataset.augmentation.mix_up_on = True
__C.dataset.augmentation.mix_up_on = False
__C.dataset.augmentation.mix_up_frequency = 1.0
# __C.dataset.augmentation.mix_up_alpha = 2.0
__C.dataset.augmentation.mix_up_alpha = 0.5

##################################
# regularization parameters
##################################

# regularization parameters
__C.regularization = {}

# regularization: label smoothing parameters
__C.regularization.label_smoothing = {}

# regularization: label smoothing on
__C.regularization.label_smoothing.on = True

# regularization: label smoothing epsilon 
__C.regularization.label_smoothing.epsilon = 0.1

####################################
# training lossd
####################################

__C.loss = {}

# the loss name, support ['softmax','focal']
# __C.loss.name = 'softmax'
__C.loss.name = 'focal'

# the weight matrix for each class in focal loss, including background class
# __C.loss.obj_weight = np.array([[1/9, 0], [0, 8/9]])
__C.loss.obj_weight = None

# the gamma parameter in focal loss
__C.loss.focal_gamma = 2

####################################
# training sampler
####################################

__C.sampler = {}

# the sampler name, support ['none','balanced']
__C.sampler.name = 'none'
# __C.sampler.name = 'balanced'

#####################################
# net
#####################################

__C.net = {}

# the network name
# __C.net.class_name = "Cnn14"
# __C.net.model_name = 'Cnn'
__C.net.class_name = "SpeechResModel"
# __C.net.model_name = 'res15_basic'
__C.net.model_name = 'res15_stochastic_depth'


######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
# __C.train.num_epochs = 16000
# __C.train.num_epochs = 8000
# __C.train.num_epochs = 4000
# __C.train.num_epochs = 100
# __C.train.num_epochs = 3
__C.train.num_epochs = 1

# the number of samples in a batch
# __C.train.batch_size = 2048
# __C.train.batch_size = 1024
# __C.train.batch_size = 128
# __C.train.batch_size = 64
__C.train.batch_size = 16
# __C.train.batch_size = 1

# the number of threads for IO
# __C.train.num_threads = 64
# __C.train.num_threads = 16
__C.train.num_threads = 1

# the number of batches to update loss curve
__C.train.plot_snapshot = 5

# the number of epochs to save model
# __C.train.save_epochs = 25
__C.train.save_epochs = 1


######################################
# learning rate parameters
######################################

# learning rate = lr*gamma**(epoch//step_size)
# __C.train.lr = 1e-3
__C.train.lr = 1e-4
# __C.train.lr = 1e-5

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

# SGD, Adam weight decay
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
__C.debug.num_processing = 64

# random seed used in training
__C.debug.seed = 0