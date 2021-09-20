from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

__C.general.data_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/KwsEnglishDataset/"
__C.general.sub_data_dir = []

# data version
# __C.general.version = "1.5"     # 重庆人录制唤醒词
# __C.general.version = "1.5.10"  # 10% 重庆人录制唤醒词
# __C.general.version = "1.5.50"  # 50% 重庆人录制唤醒
__C.general.version = "1.6"     # + 深圳人录制唤醒词（测试 tts 精度）
# __C.general.version = "1.7"
# __C.general.version = "1.8"     # tts（LibriSpeech）
# __C.general.version = "1.9.100" # tts（LibriSpeech）+ 100% 重庆人录制唤醒词
# __C.general.version = "1.9.10"  # tts（LibriSpeech）+ 10% 重庆人录制唤醒词
# __C.general.version = "1.9.50"  # tts（LibriSpeech）+ 50% 重庆人录制唤醒词
# __C.general.version = "1.10"    # tts（深圳口音生成）
# __C.general.version = "1.10.100"# tts（深圳口音生成）+ 100% 重庆人录制唤醒词

# data date
__C.general.date = "03312021"
# __C.general.date = "07162021"
# __C.general.date = "08192021"
# __C.general.date = "08272021"
# __C.general.date = "08302021"

# data path
# __C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.5_03312021/total_data_files.csv"
# __C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.5.10_03312021/total_data_files.csv"
# __C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.5.50_03312021/total_data_files.csv"
__C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.6_07162021/total_data_files.csv"
# __C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.7_08192021/total_data_files.csv"
# __C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_tts_2s_1.8_08272021/total_data_files.csv"
# __C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_tts_2s_1.9_08302021/total_data_files.csv"
# __C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_tts_2s_1.9.10_08302021/total_data_files.csv"
# __C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_tts_2s_1.9.50_08302021/total_data_files.csv"
# __C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_sztts_2s_1.10_08302021/total_data_files.csv"
# __C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_sztts_2s_1.10.100_08302021/total_data_files.csv"

# background noise path
# __C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.5_03312021/background_noise_files.csv"
# __C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.5.10_03312021/background_noise_files.csv"
# __C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.5.50_03312021/background_noise_files.csv"
__C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.6_07162021/background_noise_files.csv"
# __C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.7_08192021/background_noise_files.csv"
# __C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_tts_2s_1.8_08272021/background_noise_files.csv"
# __C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_tts_2s_1.9_08302021/background_noise_files.csv"
# __C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_tts_2s_1.9.10_08302021/background_noise_files.csv"
# __C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_tts_2s_1.9.50_08302021/background_noise_files.csv"
# __C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_sztts_2s_1.10_08302021/background_noise_files.csv"
# __C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_sztts_2s_1.10.100_08302021/background_noise_files.csv"

# test after save pytorch model
__C.general.is_test = True

# the output of training models and logging files
# __C.general.save_dir = "/mnt/huanyuan/model/kws_english_test"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_1_0_res15_fbankcpu_03222021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_1_1_res15_fbankcpu_03222021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_1_2_res15_fbankcpu_03222021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_1_3_res15_fbankcpu_03222021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_1_4_res15_fbankcpu_03222021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_1_5_res15_fbankcpu_03222021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_1_6_res15_fbankcpu_03222021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_2_1_tc-resnet14-amba_fbankcpu_kd_03222021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_2_2_tc-resnet14-amba_fbankcpu_kd_03222021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_2_3_tc-resnet14-amba_fbankcpu_dml_04012021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_2_4_tc-resnet14-amba_fbankcpu_kd_04012021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_2_5_tc-resnet14-amba_fbankcpu_kd_07162021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_wntts_1_0_res15_fbankcpu_07162021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_wntts_1_0_10_res15_fbankcpu_07162021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_tts_1_0_res15_fbankcpu_07162021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_tts_1_1_res15_fbankcpu_07162021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_tts_1_1_10_res15_fbankcpu_07162021/"
__C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_tts_1_1_50_res15_fbankcpu_07162021/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_sztts_1_1_res15_fbankcpu_07162021/"

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

# Number of audio samples per second
__C.dataset.sample_rate = 16000

# Length of each audio clip to be analyzed
__C.dataset.clip_duration_ms = 2000

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

# input size of training data (w, h), whether input size is a multiple of 16, unit: voxel
# __C.dataset.h_alignment = True, [hisi], 模型需要图像输入长度为 16 的倍数
# __C.dataset.h_alignment = False, [amba, novt]
__C.dataset.w_alignment = False
__C.dataset.h_alignment = False

# input size of training data (w, h), unit: voxel
__C.dataset.data_size = [64, 196]


##################################
# label parameters
##################################

__C.dataset.label = {}

# label
__C.dataset.label.positive_label = ["activatebwc"]
# __C.dataset.label.positive_label = ["tts"]
__C.dataset.label.positive_label_chinese_name_list = [""]
__C.dataset.label.positive_label_together = False
__C.dataset.label.positive_label_together_label = ["positive"]
__C.dataset.label.negative_label = ["_silence_", "_unknown_"]
__C.dataset.label.negative_label_together = True
__C.dataset.label.negative_label_together_label = ["negative"]
__C.dataset.label.negative_label_silence = __C.dataset.label.negative_label[0]
__C.dataset.label.negative_label_unknown = __C.dataset.label.negative_label[1]
__C.dataset.label.ignore_label = ["activate", "bwc", "tts", "temp"]
# __C.dataset.label.ignore_label = ["activatebwc", "activate", "bwc", "temp"]
__C.dataset.label.label_list = __C.dataset.label.negative_label + __C.dataset.label.positive_label
__C.dataset.label.num_classes = 2

# label percentage
__C.dataset.label.silence_percentage = 25.0         # 25%
__C.dataset.label.unknown_percentage = 2000.0       # 2000%

# trian/validation/test percentage
__C.dataset.label.validation_percentage = 15.0  # 15%
__C.dataset.label.testing_percentage = 0.0     # 0%


##################################
# data augmentation parameters
##################################

# data augmentation parameters
__C.dataset.augmentation = {}

# based on audio waveform: on
__C.dataset.augmentation.on = True
# __C.dataset.augmentation.on = False

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

# Time shift enhancement multiple of negative samples, which is effective for advanced prediction and lag prediction
__C.dataset.augmentation.time_shift_multiple = 10
# __C.dataset.augmentation.time_shift_multiple = 5

# based on audio waveform: on.
__C.dataset.augmentation.speed_volume_on = True
# __C.dataset.augmentation.speed_volume_on = False

# How fast the audio should be.
__C.dataset.augmentation.speed = [0.9, 1.1]

# How loud the audio should be.
__C.dataset.augmentation.volume = [0.4, 1.6]

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
__C.dataset.augmentation.spec_on = True
# __C.dataset.augmentation.spec_on = False
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
__C.regularization.label_smoothing.on = False

# regularization: label smoothing epsilon 
__C.regularization.label_smoothing.epsilon = 0.1


####################################
# training loss
####################################

__C.loss = {}

# the loss name, support ['softmax','focal']
# __C.loss.name = 'softmax'
__C.loss.name = 'focal'

# the weight matrix for each class in focal loss, including background class
__C.loss.obj_weight = np.array([[1/9, 0], [0, 8/9]])

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
__C.net.class_name = "SpeechResModel"
# __C.net.model_name = 'cnn-trad-pool2'
# __C.net.model_name = 'cnn-one-fstride1'
# __C.net.model_name = 'cnn-tpool2'
__C.net.model_name = 'res15'
# __C.net.model_name = 'res15-narrow'
# __C.net.model_name = 'res15-narrow-amba'
# __C.net.model_name = 'res15-narrow-novt'
# __C.net.model_name = 'res15_stochastic_depth'
# __C.net.model_name = 'res8'
# __C.net.model_name = 'res8-narrow'
# __C.net.model_name = 'lstm-avg'
# __C.net.model_name = 'lstm-attention'
# __C.net.model_name = 'crnn-avg'
# __C.net.model_name = 'crnn-attention'
# __C.net.model_name = 'tc-resnet8'
# __C.net.model_name = 'tc-resnet14'
# __C.net.model_name = 'tc-resnet8-dropout'
# __C.net.model_name = 'tc-resnet14-dropout'
# __C.net.model_name = 'tc-resnet18-dropout'
# __C.net.model_name = 'tc-resnet14-amba-novt-196'

######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
# __C.train.num_epochs = 16000
# __C.train.num_epochs = 8000
# __C.train.num_epochs = 4000
# __C.train.num_epochs = 100
__C.train.num_epochs = 1

# the number of samples in a batch
# __C.train.batch_size = 2048
# __C.train.batch_size = 1024
# __C.train.batch_size = 128
# __C.train.batch_size = 64
# __C.train.batch_size = 16
# __C.train.batch_size = 8
__C.train.batch_size = 1

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
# __C.train.optimizer = 'SGD'
__C.train.optimizer = 'Adam'

# SGD momentum
__C.train.momentum = 0.9

# SGD, Adam weight decay
# __C.train.weight_decay = 0.0
__C.train.weight_decay = 0.0001

# the beta in Adam optimizer
__C.train.betas = (0.9, 0.999)


######################################
# scheduler parameters
######################################

# only with optimizer: SGD
# scheduler, support StepLR and CosineAnnealingWarmRestarts
# __C.train.scheduler = 'StepLR'
__C.train.scheduler = 'CosineAnnealingWarmRestarts'

# StepLR step_size
__C.train.lr_step_size = 1000

# StepLR lr_gamma
__C.train.lr_gamma = 0.1

# CosineAnnealingWarmRestarts T_0
__C.train.T_0 = 2

# CosineAnnealingWarmRestarts T_mult
__C.train.T_mult = 2


######################################
# debug parameters
######################################

__C.debug = {}

# whether to save input images
# __C.debug.save_inputs = True
__C.debug.save_inputs = False

# the number of processing for save input images
__C.debug.num_processing = 64
# __C.debug.num_processing = 16

# random seed used in training
__C.debug.seed = 0


##################################
# test parameters
##################################

__C.test = {}

# the number of testing epochs
# __C.test.model_epoch = -1
__C.test.model_epoch = 200
# __C.test.model_epoch = 500
# __C.test.model_epoch = 999

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