from easydict import EasyDict as edict
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/TTS')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech/TTS')
from dataset.text.symbols import *

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

# __C.general.dataset_list = ['Aishell3']
# __C.general.dataset_path_dict = {
#                                 "Aishell3": "/mnt/huanyuan/data/speech/asr/Chinese/Aishell3/", 
#                                 "Aishell3_training": "/mnt/huanyuan/data/speech/asr/Chinese/Aishell3/train/wav", 
#                                 "Aishell3_testing": "/mnt/huanyuan/data/speech/asr/Chinese/Aishell3/test/wav", 
#                                 }
# __C.general.mutil_speaker = True

__C.general.dataset_list = ['BZNSYP']
__C.general.dataset_path_dict = {
                                "BZNSYP": "/mnt/huanyuan/data/speech/asr/Chinese/BZNSYP/", 
                                "BZNSYP_training": "/mnt/huanyuan/data/speech/asr/Chinese/BZNSYP/Wave", 
                                "BZNSYP_testing": None, 
                                }
__C.general.mutil_speaker = False

# data path
__C.general.data_dir = "/mnt/huanyuan2/data/speech/tts/Chinese_dataset/"

# the output of training models and logging files
__C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/test/"
# __C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_1_1_10232021/"
# __C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_finetune_1_2_10232021/"
# __C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_tacotron_singlespeaker_guaiding_4_2_10292021/"

# test after save pytorch model
__C.general.is_test = True

# finetune model
# __C.general.finetune_on = True
__C.general.finetune_on = False

# 模型加载方式，[0: 方式一, 1: 方式二]
__C.general.load_mode_type = 0

# 方式一：加载模型训练过程中保存模型
__C.general.finetune_model_dir = ""
__C.general.finetune_epoch = -1
# 方式二：加载其他模型结构
__C.general.finetune_model_path = "/mnt/huanyuan2/model/tts/pretrained/tacotron/pretrain_model/parameter.pkl"
__C.general.finetune_model_state = 'model_state'
# __C.general.finetune_ignore_key_list = ['module.encoder.embedding.weight']
__C.general.finetune_ignore_key_list = [
                                        'module.encoder.embedding.weight',
                                        'module.encoder_proj.weight',
                                        'module.decoder.attn_rnn.weight_ih',
                                        'module.decoder.rnn_input.weight',
                                        'module.decoder.stop_proj.weight'
                                        ]
# __C.general.finetune_model_path = "/mnt/huanyuan2/model/tts/pretrained/tacotron2/pretrain_model/parameter.pkl"
# __C.general.finetune_model_state = 'state_dict'
# __C.general.finetune_ignore_key_list = ['module.embedding.weight']
# __C.general.finetune_ignore_key_list = [
#                                         'module.embedding.weight', 
#                                         'module.decoder.attention_rnn.weight_ih', 
#                                         ]

# set certain epoch to continue training, set -1 to train from scratch
__C.general.resume_epoch = -1

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
# guiding model parameters
##################################

__C.guiding_model = {}

# __C.guiding_model.on = True
__C.guiding_model.on = False

# guide model 用于引导模型，学会 attention 信息
__C.guiding_model.config_file = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_1_1_10232021/tts_config_chinese_sv2tts.py"
__C.guiding_model.model_name = "/home/huanyuan/code/demo/Speech/TTS/network/sv2tts/tacotron.py"
__C.guiding_model.class_name = 'Tacotron'
__C.guiding_model.model_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_1_1_10232021/"
__C.guiding_model.epoch = -1

##################################
# speaker verification parameters
##################################

__C.speaker_verification = {}

__C.speaker_verification.config_file = "/mnt/huanyuan2/model/sv/Chinese_TI_SV/ti_sv_1_1_basic_10122021/sv_config_chinese_TI_SV.py"
__C.speaker_verification.model_name = "/home/huanyuan/code/demo/Speech/SV/network/basic.py"
__C.speaker_verification.class_name = 'SpeakerEncoder'
# 模型加载方式，[0: 方式一, 1: 方式二]
__C.speaker_verification.load_mode_type = 0
# 方式一：模型训练过程中，保存模型
__C.speaker_verification.model_dir = "/mnt/huanyuan2/model/sv/Chinese_TI_SV/ti_sv_1_1_basic_10122021"
__C.speaker_verification.epoch = -1
# 方式二：加载其他模型结构
__C.speaker_verification.model_path = ""
__C.speaker_verification.ignore_key_list = []

# feedback 模式：反馈约束的多说话人语音合成
# [tf_multispeakerTTS_fc](https://github.com/caizexin/tf_multispeakerTTS_fc)
# 仅用于多说话人合成任务，__C.general.mutil_speaker = True
# __C.speaker_verification.feedback_on = True
__C.speaker_verification.feedback_on = False

__C.speaker_verification.embed_loss_scale = 1.0

# loss function, support ["cos", "mse"]
__C.speaker_verification.embed_loss_func = 'cos'
# __C.speaker_verification.embed_loss_func = 'mse'


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of input channel, currently only support 1 channel input
__C.dataset.input_channel = 1

# Number of audio samples per second
__C.dataset.sample_rate = 16000

# Duration of frequency analysis window
__C.dataset.window_size_ms = 50.0

# How far to move in time between frequency windows
__C.dataset.window_stride_ms = 12.5

# How the spectrogram is processed to produce features, support ["fbank", "fbank_log", "fbank_log_manual", "pcen", "fbank_cpu"]
__C.dataset.preprocess = "fbank_log_manual"

# How many bins to use for the Mel feature
__C.dataset.feature_bin_count = 80

# How many nfilt to use for the Mel feature, only support preprocess ["fbank_cpu"]
__C.dataset.nfilt = 80

# fmin, only support preprocess ["fbank_log", "fbank_log_manual"]
# Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
__C.dataset.fmin = 55

# fmax, only support preprocess ["fbank_log", "fbank_log_manual"]
# To be increased/reduced depending on data.
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


##################################
# data augmentation parameters
##################################

# data augmentation parameters
__C.dataset.augmentation = {}

# based on audio waveform: on
# __C.dataset.augmentation.on = True
__C.dataset.augmentation.on = False

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

# based on audio waveform: on.
# __C.dataset.augmentation.speed_volume_on = True
__C.dataset.augmentation.speed_volume_on = False

# How fast the audio should be.
__C.dataset.augmentation.speed = [0.9, 1.1]

# How loud the audio should be.
__C.dataset.augmentation.volume = [1.0, 1.0]

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
# __C.dataset.augmentation.spec_on = True
__C.dataset.augmentation.spec_on = False
__C.dataset.augmentation.F = 5
__C.dataset.augmentation.T = 10
__C.dataset.augmentation.num_masks = 2


#####################################
# net
#####################################

__C.net = {}

# the network name
__C.net.model_name = "/home/huanyuan/code/demo/Speech/TTS/network/sv2tts/tacotron.py"
__C.net.class_name = "Tacotron"
# r frames
__C.net.r = 2

# # the network name
# __C.net.model_name = "/home/huanyuan/code/demo/Speech/TTS/network/sv2tts/tacotron2.py"
# __C.net.class_name = "Tacotron2"
# # r frames
# # __C.net.r = 1
# __C.net.r = 2

# speaker embedding 
# # SV2TTS, 多说话人
# __C.net.speaker_embedding_size = 256
# 单说话人
__C.net.speaker_embedding_size = 0

######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
# __C.train.num_epochs = 100
__C.train.num_epochs = 10

# the number of samples in a batch
# __C.train.batch_size = 16
__C.train.batch_size = 2

# the number of threads for IO
__C.train.num_threads = 1

# the number of batches to show log
__C.train.show_log = 5

# the number of batches to update loss curve
__C.train.plot_snapshot = 1

# the number of epochs to save model
__C.train.save_epochs = 1


######################################
# learning rate parameters
######################################

# learning rate = lr*gamma**(epoch//step_size)
__C.train.lr = 1e-3
# __C.train.lr = 1e-4

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


####################################
# training loss
####################################

__C.loss = {}

__C.loss.name = 'None'

# EMA: expontential moving average on
# EMA: https://github.com/ProLing1994/pytorch-loss/blob/master/ema.py
# __C.loss.ema_on = True
__C.loss.ema_on = False

# the alpha parameter in EMA: each parameter p should be computed as p_hat = alpha * p + (1. - alpha) * p_hat
__C.loss.ema_alpha = 0.995


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
