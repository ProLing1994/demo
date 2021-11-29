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
__C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/test/"
# __C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_1_1_10232021/"
# __C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_finetune_1_2_10232021/"
# __C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_tacotron_singlespeaker_guaiding_4_2_10292021/"
# __C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron_BahdanauAttention_singlespeaker_prosody_py_1_0_11092021/"
# __C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron_lsa_singlespeaker_prosody_py_2_1_11092021/"
# __C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_singlespeaker_prosody_py_1_0_11102021/"
# __C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_mutilspeaker_prosody_py_1_0_11102021/"
# __C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_mutilspeaker_prosody_py_1_1_11102021/"
# __C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_mutilspeaker_prosody_py_1_2_11102021/"
# __C.general.save_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_mutilspeaker_prosody_py_1_3_11102021/"

# test after save pytorch model
__C.general.is_test = True

# finetune model
# __C.general.finetune_on = True
__C.general.finetune_on = False

# 模型加载方式，[0: 根据文件目录查找, 1: 模型加载，指定文件路径]
__C.general.load_mode_type = 0

# 方式一：模型加载，根据文件目录查找
__C.general.finetune_model_dir = "/mnt/huanyuan2/model/tts/chinese_tts/sv2tts_chinese_new_tacotron2_mutilspeaker_prosody_py_1_0_11102021/"
__C.general.finetune_epoch_num = -1
__C.general.finetune_sub_folder_name = 'checkpoints'
# 方式二：模型加载，指定文件路径
__C.general.finetune_model_path = ""

__C.general.finetune_state_name = 'state_dict'
__C.general.finetune_ignore_key_list = []
# __C.general.finetune_ignore_key_list = ['module.encoder.embedding.weight']
# __C.general.finetune_ignore_key_list = [
#                                         'module.decoder.attention_rnn.rnn_cell.weight_ih', 
#                                         'module.decoder.memory_layer.weight',
#                                         'module.decoder.decoder_rnn.weight_ih',
#                                         'module.decoder.mel_proj.weight',
#                                         'module.decoder.stop_proj.weight',
#                                         ]

# module 字段添加，[0: 不添加字段, 1: 去除 module 字段, 2: 添加 module 字段]
__C.general.finetune_add_module_type = 0

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
# guiding attn mask parameters
##################################

__C.guided_attn = {}

__C.guided_attn.on = True
# __C.guided_attn.on = False

__C.guided_attn.attn_sigma = 0.4
__C.guided_attn.attn_lambda = 10.0

# 逗号，添加空白时长
__C.guided_attn.speacing_commas = 0.15
# 句号，添加空白时长
__C.guided_attn.speacing_periods = 0.20


##################################
# speaker verification parameters
##################################

__C.speaker_verification = {}

# 是否加载说话人验证模型
# __C.speaker_verification.on = True
__C.speaker_verification.on = False

__C.speaker_verification.config_file = "/mnt/huanyuan2/model/sv/Chinese_TI_SV/ti_sv_1_1_basic_10122021/sv_config_chinese_TI_SV.py"
__C.speaker_verification.model_name = "/home/huanyuan/code/demo/Speech/SV/network/basic.py"
__C.speaker_verification.class_name = 'SpeakerEncoder'

# 模型加载方式，[0: 根据文件目录查找, 1: 模型加载，指定文件路径]
__C.speaker_verification.load_mode_type = 0

# 方式一：模型加载，根据文件目录查找
__C.speaker_verification.model_dir = "/mnt/huanyuan2/model/sv/Chinese_TI_SV/ti_sv_1_1_basic_10122021"
__C.speaker_verification.epoch_num = -1
__C.speaker_verification.sub_folder_name ='checkpoints'
# 方式二：模型加载，指定文件路径
__C.speaker_verification.model_path = ""

__C.speaker_verification.state_name = 'state_dict'
__C.speaker_verification.ignore_key_list = []

# module 字段添加，[0: 不添加字段, 1: 去除 module 字段, 2: 添加 module 字段]
__C.speaker_verification.add_module_type = 0

# feedback 模式：反馈约束的多说话人语音合成
# [tf_multispeakerTTS_fc](https://github.com/caizexin/tf_multispeakerTTS_fc)
# 仅用于多说话人合成任务，__C.dataset.mutil_speaker = True
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

# Sampling rate.
__C.dataset.sampling_rate = 16000

# Length of each audio clip to be analyzed
__C.dataset.clip_duration_ms = 1600         # 1.6s

# FFT size.
__C.dataset.fft_size = 1024

# Hop size.
__C.dataset.hop_size = 256

# Window length.
__C.dataset.win_length = 1024

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
# __C.dataset.normalize_bool = True
__C.dataset.normalize_bool = False

# allow_cache
__C.dataset.allow_cache = True
# __C.dataset.allow_cache = False

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
# __C.dataset.mutil_speaker = True
__C.dataset.mutil_speaker = False

# speaker embedding, used in tacotron_old & tacotron2_old
__C.dataset.speaker_embedding_size = 256
# __C.dataset.speaker_embedding_size = 0

# num_speakers
__C.dataset.num_speakers = 166


##################################
# data augmentation parameters
##################################

# data augmentation parameters
__C.dataset.augmentation = {}

# based on audio waveform: on
# __C.dataset.augmentation.on = True
__C.dataset.augmentation.on = False

# based on longer senteces: on.
# __C.dataset.augmentation.longer_senteces_on = True
__C.dataset.augmentation.longer_senteces_on = False

# longer senteces frequency.
__C.dataset.augmentation.longer_senteces_frequency = 0.5

# longer senteces length.
__C.dataset.augmentation.longer_senteces_length = [2, 3, 4]

# longer senteces weight.
__C.dataset.augmentation.longer_senteces_weight = [0.6, 0.2, 0.2]


#####################################
# net
#####################################

__C.net = {}

# # the network name
# __C.net.model_name = "/home/huanyuan/code/demo/Speech/TTS/network/tts/tacotron_old.py"
# __C.net.class_name = "Tacotron"
# # r frames
# __C.net.r = 2

# # the network name
# __C.net.model_name = "/home/huanyuan/code/demo/Speech/TTS/network/tts/tacotron.py"
# __C.net.class_name = "Tacotron"
# __C.net.loss_name = "TacotronLoss"
# # r frames
# __C.net.r = 2

# the network name
__C.net.model_name = "/home/huanyuan/code/demo/Speech/TTS/network/tts/tacotron2.py"
__C.net.class_name = "Tacotron2"
__C.net.loss_name = "Tacotron2Loss"
# r frames
__C.net.r = 2

# # the network name
# __C.net.model_name = "/home/huanyuan/code/demo/Speech/TTS/network/tts/tacotron2_speaker_id_embedding.py"
# __C.net.class_name = "Tacotron2"
# __C.net.loss_name = "Tacotron2Loss"
# # r frames
# __C.net.r = 2


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
