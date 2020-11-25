from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

# data folder
__C.general.data_dir = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_11032020/XiaoYuDataset_11032020"
# __C.general.data_dir = "/home/engineers/yh_rmai/data/speech/kws/xiaoyu_dataset_11032020/XiaoYuDataset_11032020"

# data version
# __C.general.version = "2.2"
__C.general.version = "3.0"

# data date
# __C.general.date = "11122020"
__C.general.date = "11192020"

# data path
# __C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_11032020/dataset_1.0_11032020/total_data_files.csv"
# __C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_11032020/dataset_2.2_11122020/total_data_files.csv"
__C.general.data_csv_path = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_11032020/dataset_3.0_11192020/total_data_files.csv"
# __C.general.data_csv_path = "/home/engineers/yh_rmai/data/speech/kws/xiaoyu_dataset_11032020/dataset_1.0_11032020/total_data_files.csv"
# __C.general.data_csv_path = "/home/engineers/yh_rmai/data/speech/kws/xiaoyu_dataset_11032020/dataset_2.0_11112020/total_data_files.csv"
# __C.general.data_csv_path = "/home/engineers/yh_rmai/data/speech/kws/xiaoyu_dataset_11032020/dataset_2.2_11122020/total_data_files.csv"
# __C.general.data_csv_path = "/home/engineers/yh_rmai/data/speech/kws/xiaoyu_dataset_11032020/dataset_3.0_11192020/total_data_files.csv"

# background noise path
# __C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_11032020/dataset_1.0_11032020/background_noise_files.csv"
# __C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_11032020/dataset_2.2_11122020/background_noise_files.csv"
__C.general.background_data_path = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_11032020/dataset_3.0_11192020/background_noise_files.csv"
# __C.general.background_data_path = "/home/engineers/yh_rmai/data/speech/kws/xiaoyu_dataset_11032020/dataset_1.0_11032020/background_noise_files.csv"
# __C.general.background_data_path = "/home/engineers/yh_rmai/data/speech/kws/xiaoyu_dataset_11032020/dataset_2.0_11112020/background_noise_files.csv"
# __C.general.background_data_path = "/home/engineers/yh_rmai/data/speech/kws/xiaoyu_dataset_11032020/dataset_2.2_11122020/background_noise_files.csv"
# __C.general.background_data_path = "/home/engineers/yh_rmai/data/speech/kws/xiaoyu_dataset_11032020/dataset_3.0_11192020/background_noise_files.csv"

# test after save pytorch model
__C.general.is_test = True

# the output of training models and logging files
# __C.general.save_dir = "/home/engineers/yh_rmai/model/kws_xiaoyu3_0_timeshift_enhancement_multiple_res15_11032020/"
# __C.general.save_dir = "/home/engineers/yh_rmai/model/kws_xiaoyu3_1_timeshift_spec_on_res15_11032020/"
__C.general.save_dir = "/mnt/huanyuan/model/kws_xiaoyu_11112020_test_crnn"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu2_0_res15_10292020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu3_0_timeshift_enhancement_multiple_res15_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu3_1_timeshift_spec_on_res15_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu3_2_timeshift_spec_off_focal_res15_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu3_3_timeshift_spec_on_focal_res15_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu3_4_timeshift_spec_on_focal_balanceweight_res15_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu3_5_timeshift_spec_on_focal_res15_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu3_6_timeshift_spec_on_mask_2_focal_res15_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu4_0_fbank_timeshift_spec_on_focal_res15_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu4_1_fbank_timeshift_up_spec_on_focal_res15_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu5_0_timeshift_spec_on_res15_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu5_1_fbank_timeshift_spec_on_res15_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu5_2_timeshift_spec_on_res15_finetune_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu5_3_fbank_timeshift_spec_on_res15_finetune_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu6_0_timeshift_spec_on_res15_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu6_1_timeshift_spec_on_res15_11192020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu6_2_timeshift_spec_on_res15_11192020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu8_1_timeshift_spec_on_crnn_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu8_2_timeshift_spec_on_crnn_avg_11032020/"
# __C.general.save_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu_mining_difficult_sample_11122020/"

# finrtune model
__C.general.finetune_on = False
__C.general.finetune_model_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu4_1_fbank_timeshift_up_spec_on_focal_res15_11032020/"
__C.general.finetune_epoch = 7999

# set certain epoch to continue training, set -1 to train from scratch
__C.general.resume_epoch = -1

# the number of GPUs used in training
# __C.general.num_gpus = 4
__C.general.num_gpus = 1

# the GPUs' id used in training
# __C.general.gpu_ids = '4, 5, 6, 7'
# __C.general.gpu_ids = '6, 7'
__C.general.gpu_ids = '0'


##################################
# data set parameters
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
# __C.dataset.preprocess = "pcen"
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

# label percentage
__C.dataset.label.silence_percentage = 50.0      # 50%
__C.dataset.label.unknown_percentage = 200.0     # 200%
__C.dataset.label.difficult_sample_mining = True
__C.dataset.label.difficult_sample_percentage = 200.0     # 200%

# trian/validation/test percentage
__C.dataset.label.validation_percentage = 10.0  # 10%
__C.dataset.label.testing_percentage = 10.0     # 10%


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

# Range to randomly shift the training audio by in time(ms).
__C.dataset.augmentation.time_shift_ms = 100.0

# Time shift enhancement multiple of negative samples, which is effective for advanced prediction and lag prediction
__C.dataset.augmentation.time_shift_multiple = 15

# How fast the audio should be, just for positive samples.
__C.dataset.augmentation.possitive_speed = '0.9,1.0,1.1'

# How loud the audio should be, just for positive samples.
__C.dataset.augmentation.possitive_volume = '0.4,0.7,1.0,1.3,1.6'

# based on audio spectrum: on
__C.dataset.augmentation.spec_on = True
__C.dataset.augmentation.F = 5
__C.dataset.augmentation.T = 30
__C.dataset.augmentation.num_masks = 1

####################################
# training lossd
####################################

__C.loss = {}

# the loss name, support ['softmax','focal']
# __C.loss.name = 'softmax'
__C.loss.name = 'focal'

# the weight matrix for each class in focal loss, including background class
__C.loss.obj_weight = np.array([[1/9, 0, 0], [0, 1/9, 0], [0, 0, 7/9]])

# the gamma parameter in focal loss
__C.loss.focal_gamma = 2

#####################################
# net
#####################################

__C.net = {}

# the network name
# __C.net.name = 'cnn-trad-pool2'
# __C.net.name = 'cnn-one-fstride1'
# __C.net.name = 'cnn-tpool2'
# __C.net.name = 'res15'
# __C.net.name = 'res15-narrow'
# __C.net.name = 'res8'
# __C.net.name = 'res8-narrow'
# __C.net.name = 'lstm-avg'
__C.net.name = 'lstm-attention'
# __C.net.name = 'crnn-avg'
# __C.net.name = 'crnn-attention'
# __C.net.name = 'crnn'


######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
# __C.train.num_epochs = 16000
# __C.train.num_epochs = 8000
# __C.train.num_epochs = 4000
__C.train.num_epochs = 1

# the number of samples in a batch
# __C.train.batch_size = 2048
# __C.train.batch_size = 1024
# __C.train.batch_size = 64
# __C.train.batch_size = 16
__C.train.batch_size = 1

# the number of threads for IO
# __C.train.num_threads = 64
__C.train.num_threads = 16
# __C.train.num_threads = 1

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

# SGD,Adam weight decay
__C.train.weight_decay = 0.0

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
