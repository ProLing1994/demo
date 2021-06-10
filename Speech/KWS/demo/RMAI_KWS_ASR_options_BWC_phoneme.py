import numpy as np

from datetime import datetime
from easydict import EasyDict as edict


__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

__C.general.window_size_ms = 1000                   # 每次送入 1s 数据
__C.general.window_stride_ms = 1000                 # 每次间隔 1s 时间
__C.general.total_time_ms = 3000                    # 算法处理时长 3s 时间

__C.general.sample_rate = 16000
__C.general.nfilt = 64                              # 计算特征中，Mel 滤波器个数
__C.general.feature_freq = 64                       # 计算特征维度
__C.general.feature_time = 96                       # 每次送入 1s 数据，对应的特征时间维度 96

# kws
# activate bwc
__C.general.kws_feature_time = 196                  # kws 网络特征时间维度
__C.general.kws_stride_feature_time = 10            # kws 每间隔 10 个 feature_time 进行一次检索, 对应滑窗 100 ms，共检测 10 次
__C.general.kws_detection_threshold = 0.5           # kws 检测阈值 0.5
__C.general.kws_detection_number_threshold = 0.5    # kws 计数阈值 0.5
__C.general.kws_suppression_counter = 3             # kws 激活后抑制时间 3s

# asr
__C.general.language_id = 1			                # 0： chinese  1： english
__C.general.decode_id = 1			                # 0： greedy  1： beamsearch
__C.general.asr_feature_time = 296                  # asr 网络特征时间维度，与语音特征容器长度相同
__C.general.asr_suppression_counter = 2             # asr 激活后抑制时间，间隔 2s 执行一次 asr 检测

# # robust
# __C.general.kws_list = ['start_record', 'stop_record', 'mute_audio', 'unmute_audio', 'shot_fire', 'freeze', 'drop_gun', 'keep_hand', 'put_hand', 'down_ground']
# __C.general.kws_dict = {'start_record':["STAA1RT RAH0KAO1RD", "STAA1RT RAH0KER0D", "STAA1RT RIH0KAO1RD", "STAA1RT RIH0KER0D", "STAA1R RAH0KAO1RD", "STAA1R RAH0KER0D", "STAA1R RIH0KAO1RD", "STAA1R RIH0KER0D"], 
#                         'stop_record':["STAA1P RAH0KAO1RD", "STAA1P RAH0KER0D", "STAA1P RIH0KAO1RD", "STAA1P RIH0KER0D", "STAO1 RAH0KAO1RD", "STAO1 RAH0KER0D", "STAO1 RIH0KAO1RD", "STAO1 RIH0KER0D"], 
#                         'mute_audio':["MYUW1T AA1DIY0OW2", "MYUW1 AA1DIY0OW2"], 
#                         'unmute_audio':["AH0NMYUW1T AA1DIY0OW2", "AH0NMYUW1 AA1DIY0OW2"], 
#                         'shot_fire':["SHAA1T FAY1ER0", "SHAA1T FEH1ER0", "SHAA1T FEH1", "SHAO1 FAY1ER0", "SHAO1 FEH1ER0", "SHAO1 FEH1"], 
#                         'freeze':["FRIY1Z", "FRIY1"], 
#                         'drop_gun':["DRAA1P GAH1N", "DRAA1P GAH1", "JHAA1P GAH1N", "JHAA1P GAH1", "JHOW1 GAH1N", "JHOW1 GAH1"], 
#                         'keep_hand':["KIY1P HHAE1ND", "KIY1P HHEH1ND"], 
#                         'put_hand':["PUH1T HHAE1ND", "PUH1T HHEH1ND", "PUH1 HHAE1ND", "PUH1 HHEH1ND"], 
#                         'down_ground':["GEH1T DAW1N AA1N", "GEH1 DAW1N AA1N", "GEH1 DAW1 AA1N"]}

# strict
__C.general.kws_list = ['start_record', 'stop_record', 'mute_audio', 'unmute_audio', 'shot_fire', 'freeze', 'drop_gun', 'keep_hand', 'put_hand', 'down_ground']
__C.general.kws_dict = {'start_record':["_S T AA1 R T, _R AH0 K AO1 R D IH0 NG"], 
                        'stop_record':["_S T AA1 P, _R AH0 K AO1 R D IH0 NG"], 
                        'mute_audio':["_M Y UW1 T, _AA1 D IY0 OW2"], 
                        'unmute_audio':["_AH0 N M Y UW1 T, _AA1 D IY0 OW2"], 
                        'shot_fire':["_SH AA1 T, _F AY1 ER0"], 
                        'freeze':["_F R IY1 Z"], 
                        'drop_gun':["_D R AA1 P, _Y AO1 R _G AH1 N"], 
                        'keep_hand':["_K IY1 P, _Y AO1 R _HH AE1 N D"], 
                        'put_hand':["_P UH1 T, _Y AO1 R _HH AE1 N D"], 
                        'down_ground':["_G EH1 T, _D AW1 N _AA1 N"]}

__C.general.control_kws_list = ['start_record', 'stop_record', 'mute_audio', 'unmute_audio']

# containe
__C.general.audio_container_ms = 100                # 语音数据容器中，装有音频数据 100 ms
__C.general.audio_container_time = 10               # 语音数据容器中，装有音频数据 100 ms，对应特征维度 10
__C.general.feature_container_time = 296            # 语音特征容器中，装有时间维度 296
__C.general.feature_remove_after_time = 6           # 为保证特征一致，拼接特征需要丢弃最后的时间维度 6
__C.general.feature_remove_before_time = 100        # 为保证特征一致，拼接特征需要丢弃之前的时间维度 100

# on-off
__C.general.bool_do_kws_weakup = True
# __C.general.bool_do_kws_weakup = False
__C.general.bool_do_asr = True
__C.general.bool_output_wave = True
# __C.general.bool_output_wave = False
# __C.general.bool_output_csv = True
__C.general.bool_output_csv = False
__C.general.gpu = True

# init 
__C.general.window_size_samples = int(__C.general.sample_rate * __C.general.window_size_ms / 1000)
__C.general.window_stride_samples = int(__C.general.sample_rate * __C.general.window_stride_ms / 1000)
__C.general.window_container_samples = int(__C.general.sample_rate * __C.general.audio_container_ms / 1000)
__C.general.total_time_samples = int(__C.general.sample_rate * __C.general.total_time_ms / 1000)


##################################
# model parameters
##################################

__C.model = {}

# kws
# activate bwc
__C.model.kws_model_path = "/mnt/huanyuan/model/audio_model/amba_model/kws_activatebwc_tc_resnet14/tc_resnet14_amba_2_4_04012021.caffemodel"
__C.model.kws_prototxt_path = "/mnt/huanyuan/model/audio_model/amba_model/kws_activatebwc_tc_resnet14/tc_resnet14_amba_2_4_04012021.prototxt"
__C.model.kws_net_input_name = "data"
__C.model.kws_net_output_name = "Softmax"
__C.model.kws_chw_params = "1,64,196"
__C.model.kws_transpose = True

# asr
# __C.model.asr_model_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_phoneme_16k_06032021/asr_english_phoneme_16k_64_0603.caffemodel"
# __C.model.asr_prototxt_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_phoneme_16k_06032021/asr_english_phoneme_16k_64_0603.prototxt"
__C.model.asr_model_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_phoneme_16k_06082021/asr_english_phoneme_16k_64_0608.caffemodel"
__C.model.asr_prototxt_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_phoneme_16k_06082021/asr_english_phoneme_16k_64_0608.prototxt"
__C.model.asr_net_input_name = "data"
__C.model.asr_net_output_name = "prob"
__C.model.asr_chw_params = "1,296,64"
__C.model.asr_bpe = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_phoneme_16k_06032021/asr_english_phoneme_dict.txt"
__C.model.lm_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_phoneme_16k_06032021/4gram_asr_english_phoneme.bin"


##################################
# test parameters
##################################

# 用于 RMAI_KWS_ASR_offline_API.py
__C.test = {}

# test_mode
# 0: input_wav
# 1: input_folder
__C.test.test_mode = 0

# input_Wav
# __C.test.input_wav = "/mnt/huanyuan/model/test_straming_wav/activatebwc_1_5_03312021_validation_180.wav"
__C.test.input_wav = "/mnt/huanyuan/data/speech/Recording/Daily_Record/jabra_510/test/Jabra_510_test-kws-asr_0001.wav"

# input_folder
# __C.test.input_folder = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/test_dataset/海外同事录制_0425/路边场景/场景二/"
# __C.test.input_folder = "/home/huanyuan/share/audio_data/english_wav/office_false_alarm/"
# __C.test.input_folder = "/home/huanyuan/share/audio_data/english_wav/office_false_alarm_amba/"
__C.test.input_folder = "/home/huanyuan/share/audio_data/english_wav/freeze_true/"
# __C.test.input_folder = "/mnt/huanyuan/data/speech/Recording/demo_kws_asr_online_api/2021-06-10-15-04-32"

__C.test.output_folder = "/mnt/huanyuan/data/speech/Recording/demo_kws_asr_online_api/{}".format('-'.join('-'.join(str(datetime.now()).split('.')[0].split(' ')).split(':')))