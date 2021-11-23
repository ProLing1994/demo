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
__C.general.dataset_list = ['BZNSYP']
__C.general.dataset_path_dict = {
                                "Aishell3": "/mnt/huanyuan/data/speech/asr/Chinese/Aishell3/", 
                                "Aishell3_training": "/mnt/huanyuan/data/speech/asr/Chinese/Aishell3/train/wav", 
                                "Aishell3_testing": "/mnt/huanyuan/data/speech/asr/Chinese/Aishell3/test/wav", 
                                "BZNSYP": "/mnt/huanyuan/data/speech/asr/Chinese/BZNSYP/", 
                                "BZNSYP_training": "/mnt/huanyuan/data/speech/asr/Chinese/BZNSYP/Wave", 
                                }

# data path
__C.general.data_dir = "/mnt/huanyuan2/data/speech/tts/Chinese_dataset/"

# the output of training models and logging files
__C.general.save_dir = "/mnt/huanyuan2/model/tts_vocoder/chinese_tts_vocoder/test/"

# test after save pytorch model
__C.general.is_test = True

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

# Number of audio samples per second
__C.dataset.sample_rate = 16000

# Length of each audio clip to be analyzed
__C.dataset.clip_duration_ms = 1000

# Duration of frequency analysis window
__C.dataset.window_size_ms = 50.0

# How far to move in time between frequency windows
__C.dataset.window_stride_ms = 12.5         # hop_size = 12.5 * 16000 / 1000 = 200

# How the spectrogram is processed to produce features, support ["fbank", "fbank_log", "fbank_log_manual", "pcen", "fbank_cpu"]
__C.dataset.preprocess = "fbank_log_manual"

# How many bins to use for the Mel feature
__C.dataset.feature_bin_count = 80

# How many nfilt to use for the Mel feature, only support preprocess ["fbank_cpu"]
__C.dataset.nfilt = 80

# fmin, only support preprocess ["fbank_log", "fbank_log_manual"]
__C.dataset.fmin = 55

# fmax, only support preprocess ["fbank_log", "fbank_log_manual"]
__C.dataset.fmax = 7600

# input size of training data (w, h), whether input size is a multiple of 16, unit: voxel
# __C.dataset.h_alignment = True, [hisi], 模型需要图像输入长度为 16 的倍数
# __C.dataset.h_alignment = False, [amba, novt]
__C.dataset.w_alignment = False
__C.dataset.h_alignment = False

# input size of training data (w, h), unit: voxel
__C.dataset.data_size = [80, -1]

# language
__C.dataset.language = 'chinese'

# symbols, support ["pinyin", "prosody"]
# __C.dataset.symbols = 'pinyin'
__C.dataset.symbols = 'prosody'

# symbols, support ["en: English characters", "py: Chinese Pinyin symbols"]
# __C.dataset.symbols_lang = 'en'
__C.dataset.symbols_lang = 'py'

# num_chars
__C.dataset.num_chars = len(symbols(__C.dataset.symbols_lang))

# tts_cleaner_names
__C.dataset.tts_cleaner_names = ["basic_cleaners"]

# mutil speaker
__C.dataset.mutil_speaker = True

# speaker embedding, used in tacotron_old & tacotron2_old
# __C.dataset.speaker_embedding_size = 256
__C.dataset.speaker_embedding_size = 0

# num_speakers
__C.dataset.num_speakers = 10000


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
# __C.train.num_epochs = 1000
__C.train.num_epochs = 10

# the number of samples in a batch
# __C.train.batch_size = 6
__C.train.batch_size = 1

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
# __C.train.save_epochs = 25
__C.train.save_epochs = 1


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
