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
__C.general.kws_feature_ms = 2000                   # kws 网络特征时间, 2000ms
__C.general.kws_feature_time = 196                  # kws 网络特征时间维度
__C.general.kws_stride_feature_ms = 100             # kws 每间隔 10 个 feature_time 进行一次检索, 对应滑窗 100 ms，共检测 10 次
__C.general.kws_stride_feature_time = 10            # kws 每间隔 10 个 feature_time 进行一次检索, 对应滑窗 100 ms，共检测 10 次
__C.general.kws_detection_threshold = 0.5           # kws 检测阈值 0.5
__C.general.kws_detection_number_threshold = 0.5    # kws 计数阈值 0.5
__C.general.kws_suppression_counter = 3             # kws 激活后抑制时间 3s

# asr
__C.general.language_id = 1			                # 0： chinese  1： english
__C.general.asr_feature_time = 296                  # asr 网络特征时间维度，与语音特征容器长度相同
__C.general.asr_suppression_counter = 1             # asr 激活后抑制时间，间隔 2s 执行一次 asr 检测

__C.general.asr_second_on = False                    # asr 使用 bpe 和 phoneme 两个 model
__C.general.decode_id = 1			                # 0： greedy   1： beamsearch
__C.general.match_id = 2                            # 0:  bpe      1:  phoneme_robust  2:  phoneme_strict  3:  phoneme_combine

# containe
__C.general.audio_container_ms = 100                # 语音数据容器中，装有音频数据 100 ms
__C.general.audio_container_time = 10               # 语音数据容器中，装有音频数据 100 ms，对应特征维度 10
__C.general.feature_container_time = 296            # 语音特征容器中，装有时间维度 296
__C.general.feature_remove_after_time = 6           # 为保证特征一致，拼接特征需要丢弃最后的时间维度 6
__C.general.feature_remove_before_time = 100        # 为保证特征一致，拼接特征需要丢弃之前的时间维度 100

# on-off
__C.general.bool_output_wave = True

# init 
__C.general.window_size_samples = int(__C.general.sample_rate * __C.general.window_size_ms / 1000)
__C.general.window_stride_samples = int(__C.general.sample_rate * __C.general.window_stride_ms / 1000)
__C.general.window_container_samples = int(__C.general.sample_rate * __C.general.audio_container_ms / 1000)
__C.general.total_time_samples = int(__C.general.sample_rate * __C.general.total_time_ms / 1000)

# # robust
# __C.general.kws_list = ['start_record', 'stop_record','mute_audio', 'unmute_audio']
# __C.general.kws_phoneme_dict = {'start_record':[[["_S T", "_TH"], ["AA1", "AO1", "AH1", "AE1 S"], ["R T", "R D", "R K", "R", "T"]], \
#                                                 [["_R", "_W"], ["AH0", "IH0", "IY0", "UW1", "ER0"], ["K", "P", "K W", "_HH"], ["AO1", "AA1", "AH0", "OW1"], ["R D", "R T", "R", "L D", "L T", "L", "D", "N", "T"], ["IH0 NG", "IY0", "AHO"]]],
#                                 'stop_record':[[["_S T", "_TH"], ["AA1", "AO1", "AH1", "AE1 S"], ["P", "F", "V"]], \
#                                                 [["_R", "_W"], ["AH0", "IH0", "IY0", "UW1", "ER0"], ["K", "P", "K W", "_HH"], ["AO1", "AA1", "AH0", "OW1"], ["R D", "R T", "R", "L D", "L T", "L", "D", "N", "T"], ["IH0 NG", "IY0", "AHO"]]],
#                                 'mute_audio':[[["_M", "_HH", "_Y", "_N", "_W"], ["Y", ""], ["AH0", "IH0", "IY0", "UW1", "L"], ["T", "L", "_T", "R _T", "L T", ""]], \
#                                                 [["_AA1", "_AO1", "_OW1", "_T", "_Z", "_HH"], ["AA1", "AO1", "OW1", ""], ["D", "R D", "R T", "L D", "L T", "G L", "T", "R", "L", "V"], ["IY0 OW2", "IY0 UW1", "IY0 AH0", "IY0 L", "IY0", "AH0", "ER0", "_Y UW1", "_D UW1", "_DH AH0"]]], 
#                                 'unmute_audio':[[["_AH0", "_AA1", "_AY1", "_OW1"], ["N", "M", "L", "R M", "R", ""], ["M", "W", "Y", "_M", "_HH", "_Y", "_N" "_W"], ["Y", ""], ["AH0", "IH0", "IY0", "UW1", "L"], ["T", "L", "_T", "R _T", "L T", ""]], \
#                                                 [["_AA1", "_AO1", "_OW1", "_T", "_Z", "_HH"], ["AA1", "AO1", "OW1", ""], ["D", "R D", "R T", "L D", "L T", "G L", "T", "R", "L", "V"], ["IY0 OW2", "IY0 UW1", "IY0 AH0", "IY0 L", "IY0", "AH0", "ER0", "_Y UW1", "_D UW1", "_DH AH0"]]]}
# __C.general.control_kws_list = ['start_record', 'stop_record', 'mute_audio', 'unmute_audio']

# # strict
# __C.general.kws_list = ['start_record', 'stop_record', 'mute_audio', 'unmute_audio', 'shot_fire', 'freeze', 'drop_gun', 'keep_hand', 'put_hand', 'down_ground']
# __C.general.kws_phoneme_dict = {'start_record':[[['_S'], ['T'], ['AA1'], ['R'], ['T']],\
#                                                 [['_R'], ['AH0'], ['K'], ['AO1'], ['R'], ['D'], ['IH0'], ['NG']]], 
#                                 'stop_record':[[['_S'], ['T'], ['AA1'], ['P']], \
#                                                 [['_R'], ['AH0'], ['K'], ['AO1'], ['R'], ['D'], ['IH0'], ['NG']]], 
#                                 'mute_audio':[[['_M'], ['Y'], ['UW1'], ['T']], \
#                                                 [['_AA1'], ['D'], ['IY0'], ['OW2']]], 
#                                 'unmute_audio':[[['_AH0'], ['N'], ['M'], ['Y'], ['UW1'], ['T']], \
#                                                 [['_AA1'], ['D'], ['IY0'], ['OW2']]], 
#                                 'shot_fire':[[['_SH'], ['AA1'], ['T']], \
#                                                 [['_F'], ['AY1'], ['ER0']]], 
#                                 'freeze':[[['_F'], ['R'], ['IY1'], ['Z']]], 
#                                 'drop_gun':[[['_D'], ['R'], ['AA1'], ['P']], \
#                                                 [['_Y'], ['AO1'], ['R'], ['_G'], ['AH1'], ['N']]], 
#                                 'keep_hand':[[['_K'], ['IY1'], ['P']], \
#                                                 [['_Y'], ['AO1'], ['R'], ['_HH'], ['AE1'], ['N'], ['D']]], 
#                                 'put_hand':[[['_P'], ['UH1'], ['T']], \
#                                                 [['_Y'], ['AO1'], ['R'], ['_HH'], ['AE1'], ['N'], ['D']]], 
#                                 'down_ground':[[['_G'], ['EH1'], ['T']], \
#                                                 [['_D'], ['AW1'], ['N'], ['_AA1'], ['N']]]}
# __C.general.kws_phoneme_param_dict = {'start_record': {"verb_socres_threshold": -2.3},                    
#                                                 'stop_record': {"verb_socres_threshold": -2.3},
#                                                 'mute_audio': {"verb_socres_threshold": -2.3}, 
#                                                 'unmute_audio': {"verb_socres_threshold": -2.3},
#                                                 'shot_fire': {"verb_socres_threshold": -0.2}, 
#                                                 'freeze': {"verb_socres_threshold": -0.2}, 
#                                                 'drop_gun': {"verb_socres_threshold": -0.7}, 
#                                                 'keep_hand': {"verb_socres_threshold": -0.7}, 
#                                                 'put_hand': {"verb_socres_threshold": -0.7}, 
#                                                 'down_ground': {"verb_socres_threshold": -0.7}}
# __C.general.control_kws_list = ['start_record', 'stop_record', 'mute_audio', 'unmute_audio']

# strict simple
__C.general.kws_list = ['start_record', 'stop_record', 'mute_audio', 'unmute_audio', 'shot_fire', 'freeze', 'drop_gun', 'keep_hand', 'put_hand', 'down_ground']
__C.general.kws_phoneme_dict = {'start_record':[['_S', 'T', 'AA1', 'R', 'T'],\
                                                ['_R', 'AH0', 'K', 'AO1' , 'R' , 'D' , 'IH0' , 'NG']], 
                                'stop_record':[['_S' , 'T' , 'AA1' , 'P'], \
                                                ['_R' , 'AH0' , 'K' , 'AO1' , 'R' , 'D' , 'IH0' , 'NG']], 
                                'mute_audio':[['_M' , 'Y' , 'UW1' , 'T'], \
                                                ['_AA1' , 'D' , 'IY0' , 'OW2']], 
                                'unmute_audio':[['_AH0' , 'N' , 'M' , 'Y' , 'UW1' , 'T'], \
                                                ['_AA1' , 'D' , 'IY0' , 'OW2']], 
                                'shot_fire':[['_SH' , 'AA1' , 'T'], \
                                                ['_F' , 'AY1' , 'ER0']], 
                                'freeze':[['_F' , 'R' , 'IY1' , 'Z']], 
                                'drop_gun':[['_D' , 'R' , 'AA1' , 'P'], \
                                                ['_Y' , 'AO1' , 'R' , '_G' , 'AH1' , 'N']], 
                                'keep_hand':[['_K' , 'IY1' , 'P'], \
                                                ['_Y' , 'AO1' , 'R' , '_HH' , 'AE1' , 'N' , 'D']], 
                                'put_hand':[['_P' , 'UH1' , 'T'], \
                                                ['_Y' , 'AO1' , 'R' , '_HH' , 'AE1' , 'N' , 'D']], 
                                'down_ground':[['_G' , 'EH1' , 'T'], \
                                                ['_D' , 'AW1' , 'N' , '_AA1' , 'N']]}
__C.general.kws_phoneme_param_dict = {'start_record': {"verb_socres_threshold": -2.3},                    
                                                'stop_record': {"verb_socres_threshold": -2.3},
                                                'mute_audio': {"verb_socres_threshold": -2.3}, 
                                                'unmute_audio': {"verb_socres_threshold": -2.3},
                                                'shot_fire': {"verb_socres_threshold": -0.2}, 
                                                'freeze': {"verb_socres_threshold": -0.2}, 
                                                'drop_gun': {"verb_socres_threshold": -0.7}, 
                                                'keep_hand': {"verb_socres_threshold": -0.7}, 
                                                'put_hand': {"verb_socres_threshold": -0.7}, 
                                                'down_ground': {"verb_socres_threshold": -0.7}}
__C.general.control_kws_list = ['start_record', 'stop_record', 'mute_audio', 'unmute_audio']

# # combine
# __C.general.kws_list = ['start_record', 'stop_record', 'mute_audio', 'unmute_audio', 'shot_fire', 'freeze', 'drop_gun', 'keep_hand', 'put_hand', 'down_ground']
# __C.general.kws_phoneme_dict = {'start_record':[[["_S T", "_TH"], ["AA1", "AO1", "AH1", "AE1 S"], ["R T", "R D", "R K", "R", "T"]], \
#                                                 [["_R", "_W"], ["AH0", "IH0", "IY0", "UW1", "ER0"], ["K", "P", "K W", "_HH"], ["AO1", "AA1", "AH0", "OW1"], ["R D", "R T", "R", "L D", "L T", "L", "D", "N", "T"], ["IH0 NG", "IY0", "AHO"]]],
#                                 'stop_record':[[["_S T", "_TH"], ["AA1", "AO1", "AH1", "AE1 S"], ["P", "F", "V"]], \
#                                                 [["_R", "_W"], ["AH0", "IH0", "IY0", "UW1", "ER0"], ["K", "P", "K W", "_HH"], ["AO1", "AA1", "AH0", "OW1"], ["R D", "R T", "R", "L D", "L T", "L", "D", "N", "T"], ["IH0 NG", "IY0", "AHO"]]],
#                                 'mute_audio':[[["_M", "_HH", "_Y", "_N", "_W"], ["Y", ""], ["AH0", "IH0", "IY0", "UW1", "L"], ["T", "L", "_T", "R _T", "L T", ""]], \
#                                                 [["_AA1", "_AO1", "_OW1", "_T", "_Z", "_HH"], ["AA1", "AO1", "OW1", ""], ["D", "R D", "R T", "L D", "L T", "G L", "T", "R", "L", "V"], ["IY0 OW2", "IY0 UW1", "IY0 AH0", "IY0 L", "IY0", "AH0", "ER0", "_Y UW1", "_D UW1", "_DH AH0"]]], 
#                                 'unmute_audio':[[["_AH0", "_AA1", "_AY1", "_OW1"], ["N", "M", "L", "R M", "R", ""], ["M", "W", "Y", "_M", "_HH", "_Y", "_N" "_W"], ["Y", ""], ["AH0", "IH0", "IY0", "UW1", "L"], ["T", "L", "_T", "R _T", "L T", ""]], \
#                                                 [["_AA1", "_AO1", "_OW1", "_T", "_Z", "_HH"], ["AA1", "AO1", "OW1", ""], ["D", "R D", "R T", "L D", "L T", "G L", "T", "R", "L", "V"], ["IY0 OW2", "IY0 UW1", "IY0 AH0", "IY0 L", "IY0", "AH0", "ER0", "_Y UW1", "_D UW1", "_DH AH0"]]], 
#                                 'shot_fire':[[['_SH'], ['AA1'], ['T']], \
#                                                 [['_F'], ['AY1'], ['ER0']]], 
#                                 'freeze':[[['_F'], ['R'], ['IY1'], ['Z']]], 
#                                 'drop_gun':[[['_D'], ['R'], ['AA1'], ['P']], \
#                                                 [['_Y'], ['AO1'], ['R'], ['_G'], ['AH1'], ['N']]], 
#                                 'keep_hand':[[['_K'], ['IY1'], ['P']], \
#                                                 [['_Y'], ['AO1'], ['R'], ['_HH'], ['AE1'], ['N'], ['D']]], 
#                                 'put_hand':[[['_P'], ['UH1'], ['T']], \
#                                                 [['_Y'], ['AO1'], ['R'], ['_HH'], ['AE1'], ['N'], ['D']]], 
#                                 'down_ground':[[['_G'], ['EH1'], ['T']], \
#                                                 [['_D'], ['AW1'], ['N'], ['_AA1'], ['N']]]}
# __C.general.kws_phoneme_param_dict = {'start_record': {"verb_socres_threshold": -2.3, "strategy" : "robust"},                    
#                                                 'stop_record': {"verb_socres_threshold": -2.3, "strategy" : "robust"},
#                                                 'mute_audio': {"verb_socres_threshold": -2.3, "strategy" : "robust"}, 
#                                                 'unmute_audio': {"verb_socres_threshold": -2.3, "strategy" : "robust"},
#                                                 'shot_fire': {"verb_socres_threshold": -0.2, "strategy" : "strict"}, 
#                                                 'freeze': {"verb_socres_threshold": -0.2, "strategy" : "strict"}, 
#                                                 'drop_gun': {"verb_socres_threshold": -0.7, "strategy" : "strict"}, 
#                                                 'keep_hand': {"verb_socres_threshold": -0.7, "strategy" : "strict"}, 
#                                                 'put_hand': {"verb_socres_threshold": -0.7, "strategy" : "strict"}, 
#                                                 'down_ground': {"verb_socres_threshold": -0.7, "strategy" : "strict"}}
# __C.general.control_kws_list = ['start_record', 'stop_record', 'mute_audio', 'unmute_audio']


##################################
# model parameters
##################################

__C.model = {}
# __C.model.bool_caffe = True
__C.model.bool_caffe = False
__C.model.bool_pytorch = True

# kws
# activate bwc
## caffe
__C.model.kws_model_path = "/mnt/huanyuan/model/audio_model/amba_model/kws_activatebwc_tc_resnet14/kws_activatebwc16k_tc_resnet14_amba_2_4_04012021/tc_resnet14_amba_2_4_04012021.caffemodel"
__C.model.kws_prototxt_path = "/mnt/huanyuan/model/audio_model/amba_model/kws_activatebwc_tc_resnet14/kws_activatebwc16k_tc_resnet14_amba_2_4_04012021/tc_resnet14_amba_2_4_04012021.prototxt"
__C.model.kws_net_input_name = "data"
__C.model.kws_net_output_name = "Softmax"
__C.model.kws_chw_params = "1,64,196"
__C.model.kws_transpose = True

## pytorch
__C.model.kws_chk_path = "/mnt/huanyuan/model/audio_model/amba_model/kws_activatebwc_tc_resnet14/kws_activatebwc16k_tc_resnet14_amba_2_5_07162021/kws_activatebwc16k_tc_resnet14_amba_2_5_07162021_checkpoint_3999.pkl"
__C.model.kws_model_name = "tc-resnet14-amba-novt-196"
__C.model.kws_class_name = "SpeechResModel"
__C.model.kws_num_classes = 2
__C.model.image_height = 196
__C.model.image_weidth = 64

# asr
## caffe
__C.model.asr_model_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_phoneme_16k_06082021/asr_english_phoneme_16k_64_0608.caffemodel"
__C.model.asr_prototxt_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_phoneme_16k_06082021/asr_english_phoneme_16k_64_0608.prototxt"
__C.model.asr_dict_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_phoneme_16k_06032021/asr_english_phoneme_dict.txt"
__C.model.asr_lm_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_phoneme_16k_06032021/4gram_asr_english_phoneme.bin"
__C.model.asr_net_input_name = "data"
__C.model.asr_net_output_name = "prob"
__C.model.asr_chw_params = "1,296,64"

## pytorch
__C.model.asr_chk_path = "/mnt/huanyuan/model/audio_model/amba_model/asr_english/asr_english_phoneme_16k_06082021/asr_english_phoneme_16k_64_0608.pth"
__C.model.asr_model_name = "ASR_english_phoneme"
__C.model.asr_class_name = "ASR_English_Net"
__C.model.asr_num_classes = 136

##################################
# test parameters
##################################

# 用于 RMAI_KWS_ASR_offline_API.py
__C.test = {}

# input_Wav
__C.test.input_wav = "/home/huanyuan/share/audio_data/weakup_asr/weakup_bwc_asr_english/Jabra_510_test-kws-asr_0001.wav"

__C.test.output_folder = "/mnt/huanyuan/data/speech/Recording/demo_kws_asr_online_api/{}".format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))