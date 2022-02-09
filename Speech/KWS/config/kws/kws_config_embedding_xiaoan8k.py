from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

__C.general.data_dir = "/mnt/huanyuan2/data/speech/kws/xiaoan_dataset/experimental_dataset/XiaoAnDataset/"
__C.general.sub_data_dir = []

# data version
# __C.general.version = "2.1"     # 小安小安音频 + 困难样本挖掘（运用 AmSoftmax 思想，增大正负样本之间的差异），3s 音频
# __C.general.version = "3.1"     # 小安小安音频 + 困难样本挖掘（运用 AmSoftmax 思想，增大正负样本之间的差异），2s 音频
# __C.general.version = "3.2"     # 8k 小安小安音频(+ small voice) + 8k 负样本音频 + 困难样本挖掘，2s 音频
# __C.general.version = "3.3"     # 清洗小安小安脏数据（包含小声音频） 8k 小安小安音频(+ small voice) + 8k 负样本音频 + 困难样本挖掘，2s 音频
# __C.general.version = "3.4"     # 清洗小安小安脏数据（包含小声音频） 8k 小安小安音频(+ small voice) + 8k 负样本音频 + 困难样本挖掘，2s 音频
__C.general.version = "3.5"     # 清洗小安小安脏数据（包含小声音频） 8k 小安小安音频(+ small voice) + 8k 负样本音频 + 困难样本挖掘，2s 音频 + 增加静音比例

# data date
# __C.general.date = "11132021"
# __C.general.date = "12012021"
# __C.general.date = "12072021"
# __C.general.date = "12222021"
# __C.general.date = "01192022"
__C.general.date = "02082022"

# data path
# __C.general.data_csv_path = "/yuanhuan/data/speech/kws/xiaoan_dataset/experimental_dataset/dataset_xiaoan_3s_2_1_11112021/total_data_files.csv"
# __C.general.data_csv_path = "/yuanhuan/data/speech/kws/xiaoan_dataset/experimental_dataset/dataset_xiaoan_2s_3_1_12012021/total_data_files.csv"
# __C.general.data_csv_path = "/yuanhuan/data/speech/kws/xiaoan_dataset/experimental_dataset/dataset_xiaoan_2s_3_2_12072021/total_data_files.csv"
# __C.general.data_csv_path = "/yuanhuan/data/speech/kws/xiaoan_dataset/experimental_dataset/dataset_xiaoan_2s_3_3_12222021/total_data_files.csv"
# __C.general.data_csv_path = "/yuanhuan/data/speech/kws/xiaoan_dataset/experimental_dataset/dataset_xiaoan_2s_3_4_01192022/total_data_files.csv"
__C.general.data_csv_path = "/yuanhuan/data/speech/kws/xiaoan_dataset/experimental_dataset/dataset_xiaoan_2s_3_5_02082022/total_data_files.csv"

# background noise path
# __C.general.background_data_path = "/yuanhuan/data/speech/kws/xiaoan_dataset/experimental_dataset/dataset_xiaoan_3s_2_1_11112021/background_noise_files.csv"
# __C.general.background_data_path = "/yuanhuan/data/speech/kws/xiaoan_dataset/experimental_dataset/dataset_xiaoan_2s_3_1_12012021/background_noise_files.csv"
# __C.general.background_data_path = "/yuanhuan/data/speech/kws/xiaoan_dataset/experimental_dataset/dataset_xiaoan_2s_3_2_12072021/background_noise_files.csv"
# __C.general.background_data_path = "/yuanhuan/data/speech/kws/xiaoan_dataset/experimental_dataset/dataset_xiaoan_2s_3_3_12222021/background_noise_files.csv"
# __C.general.background_data_path = "/yuanhuan/data/speech/kws/xiaoan_dataset/experimental_dataset/dataset_xiaoan_2s_3_4_01192022/background_noise_files.csv"
__C.general.background_data_path = "/yuanhuan/data/speech/kws/xiaoan_dataset/experimental_dataset/dataset_xiaoan_2s_3_5_02082022/background_noise_files.csv"

# test after save pytorch model
__C.general.is_test = True
# __C.general.is_test = False

# the output of training models and logging files
# __C.general.save_dir = "/yuanhuan/model/kws/kws_xiaoan/test"
# __C.general.save_dir = "/yuanhuan/model/kws/kws_xiaoan/kws_xiaoan8k_3_2_tc-resnet14-hisi_fbankcpu_kd_11012021/"
# __C.general.save_dir = "/yuanhuan/model/kws/kws_xiaoan/kws_xiaoan8k_4_0_3s_tc-resnet14-hisi_fbankcpu_kd_11232021/"
# __C.general.save_dir = "/yuanhuan/model/kws/kws_xiaoan/kws_xiaoan8k_6_0_2s_tc_resnet14_fbankcpu_12022021/"
# __C.general.save_dir = "/yuanhuan/model/kws/kws_xiaoan/kws_xiaoan8k_6_1_2s_tc_resnet14_fbankcpu_embedding_12022021/"
# __C.general.save_dir = "/yuanhuan/model/kws/kws_xiaoan/kws_xiaoan8k_5_1_2s_res15_fbankcpu_diff_params_12022021/"
# __C.general.save_dir = "/yuanhuan/model/kws/kws_xiaoan/kws_xiaoan8k_5_2_2s_res15_fbankcpu_12072021/"
# __C.general.save_dir = "/yuanhuan/model/kws/kws_xiaoan/kws_xiaoan8k_5_3_2s_res15_fbankcpu_12232021/"
# __C.general.save_dir = "/yuanhuan/model/kws/kws_xiaoan/kws_xiaoan8k_5_4_2s_res15_fbankcpu_01192022/"
__C.general.save_dir = "/yuanhuan/model/kws/kws_xiaoan/kws_xiaoan8k_5_5_2s_res15_fbankcpu_02092022/"

# finetune model
__C.general.finetune_on = True

# 模型加载方式，[0: 根据文件目录查找, 1: 模型加载，指定文件路径]
__C.general.load_mode_type = 0

# 方式一：模型加载，根据文件目录查找
__C.general.finetune_model_dir = "/yuanhuan/model/kws/kws_xiaoan/kws_xiaoan8k_5_3_2s_res15_fbankcpu_12232021/"
__C.general.finetune_epoch_num = -1
__C.general.finetune_sub_folder_name ='checkpoints'
# 方式二：模型加载，指定文件路径
__C.general.finetune_model_path = ""

__C.general.finetune_state_name = 'state_dict'
__C.general.finetune_ignore_key_list = []
# __C.general.finetune_ignore_key_list = ['module.encoder.embedding.weight']
# module 字段添加，[0: 不添加字段, 1: 去除 module 字段, 2: 添加 module 字段]
__C.general.finetune_add_module_type = 0

# set certain epoch to continue training, set -1 to train from scratch
__C.general.resume_epoch_num = -1

# the number of GPUs used in training
__C.general.num_gpus = 2
# __C.general.num_gpus = 1

# the GPUs' id used in training
__C.general.gpu_ids = '0, 1'
# __C.general.gpu_ids = '0'

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
__C.dataset.sampling_rate = 8000

# Length of each audio clip to be analyzed
# __C.dataset.clip_duration_ms = 3000         # 3s
__C.dataset.clip_duration_ms = 2000         # 2s

# FFT size.
__C.dataset.fft_size = 256

# Hop size.
__C.dataset.hop_size = 80

# Window length.
__C.dataset.win_length = 256

# Window function.
__C.dataset.window = "hann"

# Number of mel basis.
__C.dataset.num_mels = 48

# How many nfilt to use for the Mel feature, only support preprocess ["fbank_cpu"]
__C.dataset.num_filts = 48

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

# compute mel type, support ["fbank", "fbank_log", "fbank_nopreemphasis_log_manual", "fbank_preemphasis_log_manual", "pcen", "fbank_cpu"]
__C.dataset.compute_mel_type = "fbank_cpu"

# input size of training data (w, h), whether input size is a multiple of 16, unit: voxel
# __C.dataset.h_alignment = True, [hisi], 模型需要图像输入长度为 16 的倍数
# __C.dataset.h_alignment = False, [amba, novt]
__C.dataset.w_alignment = False
__C.dataset.h_alignment = False

# input size of training data (w, h), unit: voxel
__C.dataset.data_size = [48, 196]

# allow_cache
__C.dataset.allow_cache = True
# __C.dataset.allow_cache = False


##################################
# label parameters
##################################

__C.dataset.label = {}

# label
# __C.dataset.label.positive_label = ["xiaoanxiaoan_8k"]
__C.dataset.label.positive_label = ["xiaoanxiaoan_8k", "xiaoanxiaoan_8k_small_voice"]
__C.dataset.label.positive_label_chinese_name_list = [""]
__C.dataset.label.positive_label_together = True
__C.dataset.label.positive_label_together_label = ["positive"]
__C.dataset.label.negative_label = ["_silence_", "_unknown_"]
__C.dataset.label.negative_label_together = True
__C.dataset.label.negative_label_together_label = ["negative"]
__C.dataset.label.negative_label_silence = __C.dataset.label.negative_label[0]
__C.dataset.label.negative_label_unknown = __C.dataset.label.negative_label[1]
__C.dataset.label.ignore_label = ['movie', 'xiaoanxiaoan_8k_once', 
                                    'nihaoxiaoan_8k', 'nihaoxiaoan_16k', 'truck_truckidling_noise_16k',
                                    'xiaoanxiaoan_16k', 'xiaoanxiaoan_16k_small_voice', 'xiaoanxiaoan_16k_once',
                                    'zanghuayulu_16k']
__C.dataset.label.label_list = __C.dataset.label.negative_label + __C.dataset.label.positive_label
__C.dataset.label.num_classes = 2

# label percentage
__C.dataset.label.silence_percentage = 50.0        # 50%
__C.dataset.label.unknown_percentage = 2000.0      # 2000%

# trian/validation/test percentage
__C.dataset.label.validation_percentage = 15.0  # 15%
__C.dataset.label.testing_percentage = 0.0     # 0%


##################################
# data augmentation parameters
##################################

# data augmentation parameters
__C.dataset.augmentation = {}

# based on audio waveform: on.
__C.dataset.augmentation.on = True
# __C.dataset.augmentation.on = False

# How many of the training samples have background noise mixed in.
__C.dataset.augmentation.background_frequency = 0.8

# How loud the background noise should be, between 0 and 1.
__C.dataset.augmentation.background_volume = 0.1

# How many of the training samples have synthetic noise mixed in.
__C.dataset.augmentation.synthetic_frequency = 0.1

# type of the synthetic noise, support ['white', 'salt_pepper'].
__C.dataset.augmentation.synthetic_type = 'white'
# __C.dataset.augmentation.synthetic_type = 'salt_pepper'

# the scale parameter in white synthetic noise
__C.dataset.augmentation.synthetic_scale = 0.001

# the prob parameter in salt pepper synthetic noise
__C.dataset.augmentation.synthetic_prob = 0.001

# Range to randomly shift the training audio by in time(ms).
# __C.dataset.augmentation.time_shift_ms = 500.0
__C.dataset.augmentation.time_shift_ms = 100.0

# Time shift enhancement multiple of negative samples, which is effective for advanced prediction and lag prediction
# __C.dataset.augmentation.time_shift_multiple = 1
__C.dataset.augmentation.time_shift_multiple = 10

# based on audio waveform: on.
__C.dataset.augmentation.speed_volume_on = True
# __C.dataset.augmentation.speed_volume_on = False

# How fast the audio should be.
__C.dataset.augmentation.speed = [0.9, 1.1]

# How loud the audio should be.
# __C.dataset.augmentation.volume = [0.4, 1.6]
__C.dataset.augmentation.volume = [0.6, 1.6]

# based on audio waveform: on.
# __C.dataset.augmentation.pitch_on = True
__C.dataset.augmentation.pitch_on = False

# How pitch the audio should be.
__C.dataset.augmentation.pitch = [-5, 5]

# based on audio waveform: on.
__C.dataset.augmentation.vad_on = True
# __C.dataset.augmentation.vad_on = False

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
__C.regularization.label_smoothing.on = True
# __C.regularization.label_smoothing.on = False

# regularization: label smoothing epsilon 
__C.regularization.label_smoothing.epsilon = 0.1


####################################
# training lossd
####################################

__C.loss = {}

# the loss method, support ['classification', 'embedding', 'classification & embedding']
__C.loss.method = 'classification'
# __C.loss.method = 'embedding'
# __C.loss.method = 'classification & embedding'

# the size of embedding in embedding method orclassification & embedding classification & embedding method
__C.loss.embedding_size = 128

# the size of embedding in embedding method orclassification & embedding classification & embedding method
__C.loss.embedding_weight = 0.001

# the loss name, support ['softmax', 'focal']
# __C.loss.name = 'softmax'
__C.loss.name = 'focal'

# the embedding loss name, support ['AmSoftmax']
__C.loss.embedding_loss_name = 'AmSoftmax'

# the number of class
__C.loss.num_classes =  __C.dataset.label.num_classes

# the weight matrix for each class in focal loss, including background class
__C.loss.obj_weight = np.array([[1/9, 0], [0, 8/9]])
# __C.loss.obj_weight = None

# the gamma parameter in focal loss
__C.loss.focal_gamma = 2

# the parameter in AmSoftmax
__C.loss.AmSoftmax_m = 0.3

# the parameter in AmSoftmax
__C.loss.AmSoftmax_s = 15

# EMA: expontential moving average on
# EMA: https://github.com/ProLing1994/pytorch-loss/blob/master/ema.py
__C.loss.ema_on = True
# __C.loss.ema_on = False

# the alpha parameter in EMA: each parameter p should be computed as p_hat = alpha * p + (1. - alpha) * p_hat
__C.loss.ema_alpha = 0.995


#####################################
# net
#####################################

__C.net = {}

# the network name
# __C.net.model_name = "/home/huanyuan/code/demo/Speech/KWS/network/tc-resnet14-amba-novt-296.py"
# __C.net.model_name = "/yuanhuan/code/demo/Speech/KWS/network/tc-resnet14-amba-novt-196.py"
__C.net.model_name = "/yuanhuan/code/demo/Speech/KWS/network/res15.py"
__C.net.class_name = "SpeechResModel"

######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
# __C.train.num_epochs = 16000
# __C.train.num_epochs = 8000
# __C.train.num_epochs = 4000
# __C.train.num_epochs = 2000
# __C.train.num_epochs = 500
__C.train.num_epochs = 300
# __C.train.num_epochs = 1

# the number of samples in a batch
# __C.train.batch_size = 2048
# __C.train.batch_size = 1024
__C.train.batch_size = 512
# __C.train.batch_size = 128
# __C.train.batch_size = 64
# __C.train.batch_size = 16
# __C.train.batch_size = 1

# the number of threads for IO
# __C.train.num_threads = 64
__C.train.num_threads = 16
# __C.train.num_threads = 4
# __C.train.num_threads = 1

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
# __C.train.lr = 1e-3
__C.train.lr = 1e-4
# __C.train.lr = 1e-5


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
# scheduler parameters
######################################

# scheduler, support [None, StepLR, CosineAnnealingWarmRestarts]
__C.train.scheduler = None
# __C.train.scheduler = 'StepLR'
# __C.train.scheduler = 'CosineAnnealingWarmRestarts'

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