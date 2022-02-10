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

# __C.general.dataset_list = ['VCC2020']
__C.general.dataset_list = ['VCC2020', 'VCC2020_reconst', 'VCC2020_cycle_reconst']
__C.general.dataset_path_dict = {
                                "VCC2020": "/mnt/huanyuan2/data/speech/asr/English/VCC2020-database/", 
                                "VCC2020_training": "/mnt/huanyuan2/data/speech/asr/English/VCC2020-database/dataset/train/", 
                                "VCC2020_testing": "/mnt/huanyuan2/data/speech/asr/English/VCC2020-database/dataset/test/", 
                                }

# state_jnt_path 
__C.general.state_jnt_path = "/yuanhuan/data/speech/vc/English_dataset/dataset_audio_normalize_hdf5/VCC2020/world/stats_jnt.h5"
__C.general.mean_name = "mean_feat_org_lf0"
__C.general.scale_name = "scale_feat_org_lf0"

# data path
__C.general.data_dir = "/yuanhuan/data/speech/vc/English_dataset"

# the output of training models and logging files
<<<<<<< HEAD
# __C.general.save_dir = "/yuanhuan/model/vc_vocoder/english_vc_vocoder/test/"
__C.general.save_dir = "/yuanhuan/model/vc_vocoder/english_vc_vocoder/wavegan_english_1_0_normalize_world_01042022/"
# __C.general.save_dir = "/yuanhuan/model/vc_vocoder/english_vc_vocoder/wavegan_english_1_1_normalize_world_cyclevae_reconst_01112022/"
# __C.general.save_dir = "/yuanhuan/model/vc_vocoder/english_vc_vocoder/wavegan_english_1_2_normalize_world_cyclevae_reconst_01112022/"
=======
# __C.general.save_dir = "/mnt/huanyuan/model/vc_vocoder/english_vc_vocoder/test/"
# __C.general.save_dir = "/mnt/huanyuan/model/vc_vocoder/english_vc_vocoder/wavegan_english_1_0_normalize_world_01042022/"
# __C.general.save_dir = "/mnt/huanyuan/model/vc_vocoder/english_vc_vocoder/wavegan_english_1_1_normalize_world_cyclevae_reconst_01112022/"
__C.general.save_dir = "/mnt/huanyuan/model/vc_vocoder/english_vc_vocoder/wavegan_english_1_2_normalize_world_cyclevae_reconst_01112022/"
>>>>>>> master

# test after save pytorch model
__C.general.is_test = True
# __C.general.is_test = False

# finetune model
__C.general.finetune_on = True
# __C.general.finetune_on = False
__C.general.finetune_model_path = "/yuanhuan/model/vc_vocoder/english_vc_vocoder/wavegan_english_1_0_normalize_world_01042022/checkpoints/chk_4500/parameter.pkl"

# resume model
# __C.general.resume_on = True
__C.general.resume_on = False
__C.general.resume_epoch_num = -1
__C.general.resume_model_path = ""

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
__C.dataset.sampling_rate = 24000

# Length of each audio clip to be analyzed
__C.dataset.clip_duration_ms = 1200

# FFT size.
__C.dataset.fft_size = 2048

# Hop size.
__C.dataset.hop_size = 120  # 5ms

# Shift length in msec (default=5)
__C.dataset.shiftms = 5 

# compute mel type, support ["world"]
__C.dataset.compute_mel_type = "world"

# normalize
__C.dataset.normalize_bool = True

# allow_cache
__C.dataset.allow_cache = True


#####################################
# net
#####################################

__C.net = {}

# the network name
__C.net.generator_model_name = "/yuanhuan/code/demo/Speech/VOCODER/network/vocoder/parallel_wavegan.py"
__C.net.generator_class_name = "ParallelWaveGANGenerator"

__C.net.discriminator_model_name = "/yuanhuan/code/demo/Speech/VOCODER/network/vocoder/parallel_wavegan.py"
__C.net.discriminator_class_name = "ParallelWaveGANDiscriminator"

__C.net.model_yaml = '/yuanhuan/code/demo/Speech/VOCODER/network/vocoder/parallel_wavegan.vc.yaml'

# load and save config
with open(__C.net.model_yaml) as f:
    __C.net.yaml = yaml.load(f, Loader=yaml.Loader)


######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
__C.train.num_epochs = 5000
# __C.train.num_epochs = 10

# the number of samples in a batch
__C.train.batch_size = 6
# __C.train.batch_size = 1

# the number of threads for IO
__C.train.num_threads = 2

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
