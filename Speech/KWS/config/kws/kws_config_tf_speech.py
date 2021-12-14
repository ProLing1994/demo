from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

# data folder
__C.general.data_dir = "/mnt/huanyuan2/data/speech/kws/tf_speech_commands/speech_commands/"
__C.general.sub_data_dir = []
# __C.general.sub_data_dir = ["/mnt/huanyuan2/data/speech/kws/tf_speech_commands/tts/sv2tts/LibriSpeech/train-clean-100/"]

# data version
# __C.general.version = "1.1"
# __C.general.version = "1.2"       # 添加 tts 合成数据
__C.general.version = "1.3"         # 数据量同 1.1，采样率为 8192，用于模型 audiomer 训练测试

# data date
# __C.general.date = "07072021"
# __C.general.date = "09302021"
__C.general.date = "12082021"

# data path
# __C.general.data_csv_path = "/mnt/huanyuan2/data/speech/kws/tf_speech_commands/dataset_1.1_07072021/total_data_files.csv"
# __C.general.data_csv_path = "/mnt/huanyuan2/data/speech/kws/tf_speech_commands/dataset_1.2_09302021/total_data_files.csv"
__C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/tf_speech_commands/dataset_1.3_12082021/total_data_files.csv"

# background noise path
# __C.general.background_data_path = "/mnt/huanyuan2/data/speech/kws/tf_speech_commands/dataset_1.1_07072021/background_noise_files.csv"
# __C.general.background_data_path = "/mnt/huanyuan2/data/speech/kws/tf_speech_commands/dataset_1.2_09302021/background_noise_files.csv"
__C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/tf_speech_commands/dataset_1.3_12082021/background_noise_files.csv"

# test after save pytorch model
__C.general.is_test = True

# the output of training models and logging files
__C.general.save_dir = "/mnt/huanyuan/model/kws/kws_speech/test"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_speech_1_0_res15_02042021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_speech_1_1_edge-speech-nets_02042021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_speech_1_2_tc-resnet8_02192021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_speech_1_3_tc-resnet14_02192021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_speech_1_4_tc-resnet8-dropout_02192021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_speech_1_5_tc-resnet14-dropout_02192021/"
# __C.general.save_dir = "/mnt/huanyuan/model/kws/kws_speech/kws_speech_2_7_res15_basic_label_smooth_ema_augmentation_tts_8000_epoch_09302021/"

# finetune model
__C.general.finetune_on = False

# set certain epoch to continue training, set -1 to train from scratch
__C.general.resume_epoch_num = -1

# the number of GPUs used in training
# __C.general.num_gpus = 4
__C.general.num_gpus = 1

# the GPUs' id used in training
# __C.general.gpu_ids = '0, 1, 2, 3'
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

# loss
# kd: https://github.com/peterliht/knowledge-distillation-pytorch
# cd: https://github.com/zhouzaida/channel-distillation
__C.knowledge_distillation.loss_name = 'kd'

# kd, alpha
__C.knowledge_distillation.alpha = 0.95

# kd, temperature
__C.knowledge_distillation.temperature = 6


##################################
# Deep Mutual Learning parameters
##################################

__C.deep_mutual_learning = {}

# Deep Mutual Learning: on
# DML: https://arxiv.org/abs/1706.00384
__C.deep_mutual_learning.on = False

# model number
__C.deep_mutual_learning.model_num = 2


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of input channel, currently only support 1 channel input
__C.dataset.input_channel = 1

# Sampling rate.
__C.dataset.sampling_rate = 8192            # 模型 audiomer 使用的采样率为 8192
# __C.dataset.sampling_rate = 16000

# Length of each audio clip to be analyzed
__C.dataset.clip_duration_ms = 1000

# FFT size.
__C.dataset.fft_size = 480

# Hop size.
__C.dataset.hop_size = 160

# Window length.
__C.dataset.win_length = 480

# Window function.
__C.dataset.window = "hann"

# Number of mel basis.
__C.dataset.num_mels = 40

# How many nfilt to use for the Mel feature, only support preprocess ["fbank_cpu"]
__C.dataset.num_filts = 40

# Minimum freq in mel basis calculation, only support preprocess ["fbank_log", "fbank_nopreemphasis_log_manual", "fbank_preemphasis_log_manual"]
# Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
__C.dataset.fmin = None

# Maximum frequency in mel basis calculation, only support preprocess ["fbank_log", "fbank_nopreemphasis_log_manual", "fbank_preemphasis_log_manual"]
# To be increased/reduced depending on data.
__C.dataset.fmax = None

# trim silence：Whether to trim the start and end of silence
# __C.dataset.trim_silence = True
__C.dataset.trim_silence = False

# Need to tune carefully if the recording is not good.
__C.dataset.trim_threshold_in_db = 60

# Frame size in trimming.
__C.dataset.trim_frame_size = 2048

# Hop size in trimming.
__C.dataset.trim_hop_size = 512

# compute mel type, support ["wave", "fbank", "fbank_log", "fbank_nopreemphasis_log_manual", "fbank_preemphasis_log_manual", "pcen", "fbank_cpu"]
# __C.dataset.compute_mel_type = "fbank_log"
__C.dataset.compute_mel_type = "wave"

# input size of training data (w, h), whether input size is a multiple of 16, unit: voxel
# __C.dataset.h_alignment = True, [hisi], 模型需要图像输入长度为 16 的倍数
# __C.dataset.h_alignment = False, [amba, novt]
__C.dataset.w_alignment = False
__C.dataset.h_alignment = False


# input size of training data (w, h), unit: voxel
__C.dataset.data_size = [40, 101]

# allow_cache
__C.dataset.allow_cache = True
# __C.dataset.allow_cache = False

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
__C.dataset.label.negative_label_together = False
__C.dataset.label.negative_label_together_label = ["negative"]
__C.dataset.label.negative_label_silence = __C.dataset.label.negative_label[0]
__C.dataset.label.negative_label_unknown = __C.dataset.label.negative_label[1]
__C.dataset.label.ignore_label = []
__C.dataset.label.label_list = __C.dataset.label.negative_label + __C.dataset.label.positive_label
__C.dataset.label.num_classes = len(__C.dataset.label.positive_label) + len(__C.dataset.label.negative_label)

# label percentage
__C.dataset.label.silence_percentage = 10.0     # 10%
__C.dataset.label.unknown_percentage = 10.0     # 10%

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

# How many of the training samples have synthetic noise mixed in.
__C.dataset.augmentation.synthetic_frequency = -1

# type of the synthetic noise, support ['white', 'salt_pepper'].
__C.dataset.augmentation.synthetic_type = 'white'
# __C.dataset.augmentation.synthetic_type = 'salt_pepper'

# the scale parameter in white synthetic noise
__C.dataset.augmentation.synthetic_scale = 0.001

# the prob parameter in salt pepper synthetic noise
__C.dataset.augmentation.synthetic_prob = 0.001

# Range to randomly shift the training audio by in time(ms).
__C.dataset.augmentation.time_shift_ms = 100.0

# Time shift enhancement multiple of negative samples, which is effective for advanced prediction and lag prediction
__C.dataset.augmentation.time_shift_multiple = 1

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

# based on audio waveform: on.
# __C.dataset.augmentation.vad_on = True
__C.dataset.augmentation.vad_on = False

# How many of the training samples have vad augmentation.
__C.dataset.augmentation.vad_frequency = 0.1

# window size of the vad. 
# Must be either 10, 20 or 30 milliseconds. This sets the granularity of the VAD. Should not need to be changed.
__C.dataset.augmentation.vad_window_length = [10, 20, 30]

# vad mode
# 0: Normal，1：low Bitrate，2：Aggressive，3：Very Aggressive
__C.dataset.augmentation.vad_mode = [0, 1, 2]

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


##################################
# regularization parameters
##################################

# regularization parameters
__C.regularization = {}

# regularization: label smoothing parameters
__C.regularization.label_smoothing = {}

# regularization: label smoothing on
# __C.regularization.label_smoothing.on = True
__C.regularization.label_smoothing.on = False

# regularization: label smoothing epsilon 
__C.regularization.label_smoothing.epsilon = 0.1


####################################
# training loss
####################################

__C.loss = {}

# the loss method, support ['classification', 'embedding', 'classification & embedding']
__C.loss.method = 'classification'

# the loss name, support ['softmax','focal']
__C.loss.name = 'softmax'
# __C.loss.name = 'focal'

# the weight matrix for each class in focal loss, including background class
__C.loss.obj_weight = np.array([[1/9, 0, 0], [0, 1/9, 0], [0, 0, 7/9]])

# the gamma parameter in focal loss
__C.loss.focal_gamma = 2

# EMA: expontential moving average on
# EMA: https://github.com/ProLing1994/pytorch-loss/blob/master/ema.py
# __C.loss.ema_on = True
__C.loss.ema_on = False

# the alpha parameter in EMA: each parameter p should be computed as p_hat = alpha * p + (1. - alpha) * p_hat
__C.loss.ema_alpha = 0.995

#####################################
# net
#####################################

__C.net = {}

# the network name
# __C.net.model_name = "/home/huanyuan/code/demo/Speech/KWS/network/bc-resnet.py"
# __C.net.class_name = "BCResNet"
__C.net.model_name = "/home/huanyuan/code/demo/Speech/KWS/network/audiomer.py"
__C.net.class_name = "AudiomerClassification"


######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
# __C.train.num_epochs = 4000
__C.train.num_epochs = 1

# the number of samples in a batch
# __C.train.batch_size = 2048
# __C.train.batch_size = 32
__C.train.batch_size = 1

# the number of threads for IO
# __C.train.num_threads = 64
__C.train.num_threads = 1

# the number of batches to show log
__C.train.show_log = 5

# the number of batches to update loss curve
__C.train.plot_snapshot = 5

# the number of epochs to save model
__C.train.save_epochs = 25


######################################
# learning rate parameters
######################################

# learning rate = lr*gamma**(epoch//step_size)
__C.train.lr = 1e-3
# __C.train.lr = 1e-4


######################################
# optimizer parameters
######################################

# optimizer, support SGD and Adam
# __C.train.optimizer = 'SGD'
__C.train.optimizer = 'Adam'

# SGD, Adam momentum
__C.train.momentum = 0.9

# SGD, Adam weight decay
# __C.train.weight_decay = 0.0
__C.train.weight_decay = 0.0001

# the beta in Adam optimizer
__C.train.betas = (0.9, 0.999)


######################################
# scheduler parameters
######################################

# scheduler, support [None, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts]
__C.train.scheduler = None
# __C.train.scheduler = 'StepLR'
# __C.train.scheduler = 'CosineAnnealingLR'
# __C.train.scheduler = 'CosineAnnealingWarmRestarts'

# StepLR step_size
__C.train.lr_step_size = 1000

# StepLR lr_gamma
__C.train.lr_gamma = 0.1

# CosineAnnealingWarmRestarts T_0
__C.train.T_0 = 2

# CosineAnnealingWarmRestarts T_mult
__C.train.T_mult = 2

# CosineAnnealingLR T_max
__C.train.T_max = 100


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

##################################
# test parameters
##################################

__C.test = {}

# the number of testing epochs
__C.test.model_epoch = -1

# mode, support [0: RecognizeCommands, 1: RecognizeCommandsCountNumber, 2:RecognizeCommandsAlign]
__C.test.method_mode = 0

# detection threshold, support [0.3,0.4,0.6,0.8,0.9,0.95]
# __C.test.detection_threshold = 0.95
__C.test.detection_threshold = 0.8

# detection number threshold, only support method_mode=1:RecognizeCommandsCountNumber
__C.test.detection_number_threshold = 0.9   # [0.5,0.75,0.9]

# detection threshold low & high, only support method_mode=2:RecognizeCommandsAlign
__C.test.detection_threshold_low = 0.1
__C.test.detection_threshold_high = __C.test.detection_threshold
    
# parameter
__C.test.timeshift_ms = 30
__C.test.average_window_duration_ms = 800    # [450,800,1500]
__C.test.minimum_count = 10
__C.test.suppression_ms = 2500               # [500, 3000]