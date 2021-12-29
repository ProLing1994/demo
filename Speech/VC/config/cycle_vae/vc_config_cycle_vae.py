from easydict import EasyDict as edict
import yaml

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

__C.general.dataset_list = ['VCC2020']
__C.general.dataset_path_dict = {
                                "VCC2020": "/mnt/huanyuan2/data/speech/asr/English/VCC2020-database/", 
                                "VCC2020_training": "/mnt/huanyuan2/data/speech/asr/English/VCC2020-database/dataset/train/", 
                                "VCC2020_testing": "/mnt/huanyuan2/data/speech/asr/English/VCC2020-database/dataset/test/", 
                                }

# data path
__C.general.data_dir = "/mnt/huanyuan/data/speech/vc/English_dataset/"

# the output of training models and logging files
__C.general.save_dir = "/mnt/huanyuan/model/vc/english_vc/test_1229/"

# test after save pytorch model
__C.general.is_test = True

# finetune model
__C.general.finetune_on = False

# set certain epoch to continue training, set -1 to train from scratch
__C.general.resume_epoch_num = -1

# the number of GPUs used in training
__C.general.num_gpus = 1

# the GPUs' id used in training
__C.general.gpu_ids = '0'

# data_parallel_mode: [0, 1]
# 0: 单机多卡，DataParallel
# 1: 单/多级多卡、分布式，DistributedDataParallel
# 2: 单机多卡，数据手动并行
## TODO：目前在训练 sv2tts 过程中，多卡运行出现异常，原因未知，bug: terminate called after throwing an instance of 'c10::Error'
__C.general.data_parallel_mode = 0


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of input channel, currently only support 1 channel input
__C.dataset.input_channel = 1

# Sampling rate
__C.dataset.sampling_rate = 24000

# Fft length (default=1024)
__C.dataset.fft_size = 2048

# Shift length in msec (default=5)
__C.dataset.shiftms = 5

# Dimension of mel-cepstrum
__C.dataset.mcep_dim = 49

# Alpha value of mel-cepstrum
__C.dataset.mcep_alpha=0.466 # 24k

# Highpass filter cutoff frequency (if 0, will not apply)
__C.dataset.highpass_cutoff = 65

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
__C.net.encoder_model_name = "/home/huanyuan/code/demo/Speech/VC/network/cycle_vae/gru_vae.py"
__C.net.encoder_class_name = "GRU_RNN_STOCHASTIC"

__C.net.decoder_model_name = "/home/huanyuan/code/demo/Speech/VC/network/cycle_vae/gru_vae.py"
__C.net.decoder_class_name = "GRU_RNN"

__C.net.model_yaml = "/home/huanyuan/code/demo/Speech/VC/network/cycle_vae/cycle_vae.yaml"

# load and save config
with open(__C.net.model_yaml) as f:
    __C.net.yaml = yaml.load(f, Loader=yaml.Loader)

######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
__C.train.num_epochs = 80

# the number of samples in a batch
__C.train.batch_size = 5

# the number of frame in a batch
__C.train.batch_frame_size = 80

# the number of threads for IO
__C.train.num_threads = 1

# the number of batches to show profiler
__C.train.show_profiler = 5

# the number of batches to update loss curve
__C.train.plot_snapshot = 5

# the number of epochs to save model
__C.train.save_epochs = 25


######################################
# optimizer parameters
######################################

__C.optimizer = {}

__C.optimizer.lr = 1e-4

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