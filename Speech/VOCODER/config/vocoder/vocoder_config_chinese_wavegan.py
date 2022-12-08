from easydict import EasyDict as edict
import sys
import yaml

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from TTS.dataset.text.symbols import *

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

# __C.general.dataset_list = ['Aishell3', 'BZNSYP']
__C.general.dataset_list = ['BZNSYP', 'BZNSYP_Tacotron2']
__C.general.dataset_path_dict = {
                                "Aishell3": "/mnt/huanyuan/data/speech/asr/Chinese/Aishell3/", 
                                "Aishell3_training": "/mnt/huanyuan/data/speech/asr/Chinese/Aishell3/train/wav", 
                                "Aishell3_testing": "/mnt/huanyuan/data/speech/asr/Chinese/Aishell3/test/wav", 
                                "BZNSYP": "/mnt/huanyuan/data/speech/asr/Chinese/BZNSYP/", 
                                "BZNSYP_training": "/mnt/huanyuan/data/speech/asr/Chinese/BZNSYP/Wave", 
                                "BZNSYP_Tacotron2": None,
                                "BZNSYP_Tacotron2_training": None,
                                }

# state_jnt_path 
__C.general.state_jnt_path = "/mnt/huanyuan/data/speech/tts/Chinese_dataset/dataset_audio_normalize_hdf5/BZNSYP_stats.h5"
__C.general.mean_name = "mean"
__C.general.scale_name = "scale"

# data path
__C.general.data_dir = "/mnt/huanyuan/data/speech/tts/Chinese_dataset/"
# __C.general.data_dir = "/yuanhuan/data/speech/tts/Chinese_dataset/"

# the output of training models and logging files
# __C.general.save_dir = "/mnt/huanyuan/model/tts_vocoder/chinese_tts_vocoder/test/"
# __C.general.save_dir = "/mnt/huanyuan/model/tts_vocoder/chinese_tts_vocoder/wavegan_chinese_singlespeaker_1_0_11232021/"
# __C.general.save_dir = "/mnt/huanyuan/model/tts_vocoder/chinese_tts_vocoder/wavegan_chinese_singlespeaker_1_1_normalize_11232021/"
# __C.general.save_dir = "/mnt/huanyuan/model/tts_vocoder/chinese_tts_vocoder/wavegan_chinese_singlespeaker_1_2_normalize_diff_feature_11292021/"
__C.general.save_dir = "/mnt/huanyuan/model/tts_vocoder/chinese_tts_vocoder/wavegan_chinese_singlespeaker_1_3_normalize_diff_feature_fineune_11292021/"

# test after save pytorch model
__C.general.is_test = True
# __C.general.is_test = False

# finetune model
# __C.general.finetune_on = True
__C.general.finetune_on = False
__C.general.finetune_model_path = "/mnt/huanyuan2/model/tts_vocoder/pretrained/wavernn/pretrain_model/parameter.pkl"

# resume model
# __C.general.resume_on = True
__C.general.resume_on = False
__C.general.resume_epoch_num = -1
__C.general.resume_model_path = "/mnt/huanyuan2/model/tts_vocoder/pretrained/wavernn/pretrain_model/parameter.pkl"

# the number of GPUs used in training
__C.general.num_gpus = 1

# the GPUs' id used in training
__C.general.gpu_ids = '0'

# data_parallel_mode: [0, 1]
# 0: 单机多卡，DataParallel
# 1: 单/多级多卡、分布式，DistributedDataParallel
__C.general.data_parallel_mode = 0


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of input channel, currently only support 1 channel input
__C.dataset.input_channel = 1

# Sampling rate.
__C.dataset.sampling_rate = 16000

# Length of each audio clip to be analyzed
__C.dataset.clip_duration_ms = 1000

# FFT size.
__C.dataset.fft_size = 1024
# __C.dataset.fft_size = 800

# Hop size.
__C.dataset.hop_size = 256
# __C.dataset.hop_size = 200

# Window length.
__C.dataset.win_length = 1024
# __C.dataset.win_length = 800

# Window function.
__C.dataset.window = "hann"

# Number of mel basis.
__C.dataset.num_mels = 80

# How many nfilt to use for the Mel feature, only support preprocess ["fbank_cpu"]
__C.dataset.num_filts = 80

# Minimum freq in mel basis calculation, only support preprocess ["fbank_log", "fbank_nopreemphasis_log_manual", "fbank_preemphasis_log_manual"]
# Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
__C.dataset.fmin = 80
# __C.dataset.fmin = 55

# Maximum frequency in mel basis calculation, only support preprocess ["fbank_log", "fbank_nopreemphasis_log_manual", "fbank_preemphasis_log_manual"]
# To be increased/reduced depending on data.
__C.dataset.fmax = 7600

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
__C.dataset.compute_mel_type = "fbank_nopreemphasis_log_manual"
# __C.dataset.compute_mel_type = "fbank_preemphasis_log_manual"

# input size of training data (w, h), whether input size is a multiple of 16, unit: voxel
# __C.dataset.h_alignment = True, [hisi], 模型需要图像输入长度为 16 的倍数
# __C.dataset.h_alignment = False, [amba, novt]
__C.dataset.w_alignment = False
__C.dataset.h_alignment = False

# input size of training data (w, h), unit: voxel
__C.dataset.data_size = [80, -1]

# normalize
__C.dataset.normalize_bool = True
# __C.dataset.normalize_bool = False

# allow_cache
__C.dataset.allow_cache = True
# __C.dataset.allow_cache = False

#####################################
# net
#####################################

__C.net = {}

# the network name
__C.net.generator_model_name = "/home/huanyuan/code/demo/Speech/VOCODER/network/vocoder/parallel_wavegan.py"
__C.net.generator_class_name = "ParallelWaveGANGenerator"

__C.net.discriminator_model_name = "/home/huanyuan/code/demo/Speech/VOCODER/network/vocoder/parallel_wavegan.py"
__C.net.discriminator_class_name = "ParallelWaveGANDiscriminator"

__C.net.model_yaml = '/home/huanyuan/code/demo/Speech/VOCODER/network/vocoder/parallel_wavegan.v1.yaml'

# load and save config
with open(__C.net.model_yaml) as f:
    __C.net.yaml = yaml.load(f, Loader=yaml.Loader)


######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
__C.train.num_epochs = 1000
# __C.train.num_epochs = 10

# the number of samples in a batch
__C.train.batch_size = 6
# __C.train.batch_size = 1

# the number of threads for IO
__C.train.num_threads = 1

# the number of batches to start train generator
__C.train.generator_train_start_steps = 0

# the number of batches to start train discriminator
__C.train.discriminator_train_start_steps = 100000

# the number of batches to show log
__C.train.show_log = 100

# the number of batches to update loss curve
__C.train.plot_snapshot = 100

# the number of epochs to save model
__C.train.save_epochs = 25
# __C.train.save_epochs = 1

####################################
# training loss
####################################

__C.loss = {}

__C.loss.generator_loss = {}
__C.loss.generator_loss.average_by_discriminators = True
__C.loss.generator_loss.loss_type = "mse"

__C.loss.discriminator_loss = {}
__C.loss.discriminator_loss.average_by_discriminators = True
__C.loss.discriminator_loss.loss_type = "mse"

# STFT LOSS SETTING
__C.loss.stft_loss = {}
__C.loss.stft_loss.on = True
__C.loss.stft_loss.fft_sizes = [1024, 2048, 512]  # List of FFT size for STFT-based loss.
__C.loss.stft_loss.hop_sizes = [120, 240, 50]     # List of hop size for STFT-based loss
__C.loss.stft_loss.win_lengths = [600, 1200, 240] # List of window length for STFT-based loss.
__C.loss.stft_loss.window = "hann_window"         # Window function for STFT-based loss

__C.loss.lambda_aux = 1.0  # Loss balancing coefficient.
__C.loss.lambda_adv = 4.0  # Loss balancing coefficient.

__C.loss.subband_stft_loss = {}
__C.loss.subband_stft_loss.on = False

__C.loss.feat_match_loss = {}
__C.loss.feat_match_loss.on = False

__C.loss.mel_loss = {}
__C.loss.mel_loss.on = False


######################################
# optimizer parameters
######################################

__C.optimizer = {}

__C.optimizer.generator_optimizer = {}
__C.optimizer.generator_optimizer.type = "RAdam"
__C.optimizer.generator_optimizer.lr = 0.0001 
__C.optimizer.generator_optimizer.betas = (0.9, 0.999)
__C.optimizer.generator_optimizer.eps = 1.0e-6
__C.optimizer.generator_optimizer.weight_decay = 0.0

__C.optimizer.generator_scheduler = {}
__C.optimizer.generator_scheduler.type = "StepLR"
__C.optimizer.generator_scheduler.step_size = 200000
__C.optimizer.generator_scheduler.gamma = 0.5

__C.optimizer.generator_grad_norm = 10                  # Generator's gradient norm.

__C.optimizer.discriminator_optimizer = {}
__C.optimizer.discriminator_optimizer.type = "RAdam"
__C.optimizer.discriminator_optimizer.lr = 0.00005
__C.optimizer.discriminator_optimizer.betas = (0.9, 0.999)
__C.optimizer.discriminator_optimizer.eps = 1.0e-6
__C.optimizer.discriminator_optimizer.weight_decay = 0.0

__C.optimizer.discriminator_scheduler = {}
__C.optimizer.discriminator_scheduler.type = "StepLR"
__C.optimizer.discriminator_scheduler.step_size = 200000
__C.optimizer.discriminator_scheduler.gamma = 0.5

__C.optimizer.discriminator_grad_norm = 1               # Discriminator's gradient norm.


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
