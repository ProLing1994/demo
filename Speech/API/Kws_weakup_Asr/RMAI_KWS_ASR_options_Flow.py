from easydict import EasyDict as edict
import os

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

__C.general.int16_max = (2 ** 15) - 1

__C.general.window_size_ms = 1000                   # 每次送入 1s 数据
__C.general.window_stride_ms = 1000                 # 每次间隔 1s 时间
__C.general.total_time_ms = 4000                    # 算法处理时长 4s 时间

__C.general.sample_rate = 16000
__C.general.nfilt = 64                              # 计算特征中，Mel 滤波器个数
__C.general.feature_freq = 64                       # 计算特征维度
__C.general.feature_time = 96                       # 每次送入 1s 数据，对应的特征时间维度 96

# vad
__C.general.vad_window_length = 30                  # In milliseconds，30 ms 音频用于 vad 计算
__C.general.vad_moving_average_width = 8            # 平滑长度，连续 8 帧平滑
__C.general.vad_max_silence_length = 6              # 利用膨胀腐蚀思想，减少空洞现象

# kws
# activate bwc
__C.general.kws_feature_time = 192                  # kws 网络特征时间维度
__C.general.kws_stride_feature_time = 10            # kws 每间隔 10 个 feature_time 进行一次检索, 对应滑窗 100 ms，共检测 10 次
__C.general.kws_detection_threshold = 0.5           # kws 检测阈值 0.5
__C.general.kws_detection_number_threshold = 0.5    # kws 计数阈值 0.5
__C.general.kws_suppression_counter = 3             # kws 激活后抑制时间 3s

# asr mandarin taxi
__C.general.language_id = 0			                # 0： chinese  1： english
__C.general.decode_id = 1			                # 0： greedy  1： beamsearch
__C.general.asr_feature_time = 296                  # asr 网络特征时间维度，与语音特征容器长度相同
__C.general.asr_suppression_counter = 2             # asr 激活后抑制时间，间隔 2s 执行一次 asr 检测
__C.general.asr_bpe_phoneme_on = False              # asr 使用 bpe 和 phoneme 两个 model

# asr vad mandarin
__C.general.asr_vad_counter_ms = 2000               # vad asr 最少检测音频 2s

# container
__C.general.audio_container_ms = 100                # 语音数据容器中，装有音频数据 100 ms
__C.general.audio_container_time = 10               # 语音数据容器中，装有音频数据 100 ms，对应特征维度 10
__C.general.feature_container_time = 296            # 语音特征容器中，装有时间维度 296
__C.general.feature_remove_after_time = 6           # 为保证特征一致，拼接特征需要丢弃最后的时间维度 6
__C.general.feature_remove_before_time = 100        # 为保证特征一致，拼接特征需要丢弃之前的时间维度 100
__C.general.vad_container_time = 4                  # vad 容器，判断连续 4s 中是否全部为静音，用于停止后续操作
__C.general.asr_vad_audio_data_ms = 6000            # 语音数据容器，用于带 vad 的 asr 识别，装有音频数据 6000ms

# chinese key words
__C.general.kws_list = [
                        '小锐小锐_唤醒', 
                        '他妈的_脏话', '傻逼_脏话', '草泥马_脏话',
                        '帮我报警_救援', '拨打紧急联系人_救援', '请求道路救援_救援',
                        ]
__C.general.kws_dict = {'小锐小锐_唤醒':'xiao rui xiao rui',
                        
                        '他妈的_脏话':'ta ma de',
                        '傻逼_脏话':'sha bi',
                        '草泥马_脏话':'cao ni ma', 

                        '帮我报警_救援':'bang wo bao jing',
                        '拨打紧急联系人_救援':'bo da jin ji lian xi ren', 
                        '请求道路救援_救援': 'qing qiu dao lu jiu yuan',
                        }
__C.general.control_kws_list = []

# init 
__C.general.window_size_samples = int(__C.general.sample_rate * __C.general.window_size_ms / 1000)
__C.general.window_stride_samples = int(__C.general.sample_rate * __C.general.window_stride_ms / 1000)
__C.general.window_container_samples = int(__C.general.sample_rate * __C.general.audio_container_ms / 1000)
__C.general.total_time_samples = int(__C.general.sample_rate * __C.general.total_time_ms / 1000)


##################################
# model parameters
##################################

__C.model = {}
# __C.model.bool_caffe = True
__C.model.bool_caffe = False
__C.model.bool_pytorch = True

# kws
# xiaorui
## caffe
__C.model.kws_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "kws_xiaorui16k_tc_resnet14_hisi_6_3_06302021.caffemodel")
__C.model.kws_prototxt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "kws_xiaorui16k_tc_resnet14_hisi_6_3_06302021.prototxt")
__C.model.kws_label = "xiaorui"
__C.model.kws_net_input_name = "data"
__C.model.kws_net_output_name = "prob"
__C.model.kws_chw_params = "1,64,192"
__C.model.kws_transpose = True

## pytorch 
__C.model.kws_chk_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "xiaorui16k_tc-resnet14_hisi_6_3_checkpoints_1999.pkl")
__C.model.kws_model_name = "tc-resnet14-amab-hisi-novt-192"
__C.model.kws_class_name = "SpeechResModel"
__C.model.kws_num_classes = 2
__C.model.image_height = 192
__C.model.image_weidth = 64

# asr
## caffe
__C.model.asr_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "?")
__C.model.asr_prototxt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "?")
__C.model.asr_dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "asr_mandarin_pinyin_408.txt")
__C.model.asr_lm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "4gram_asr_mandarin_408.bin")
__C.model.asr_net_input_name = "data"
__C.model.asr_net_output_name = "prob"
__C.model.asr_chw_params = "1,296,64"

## pytorch 
__C.model.asr_chk_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "asr_mandarin_canbin_16k_64.pth")
__C.model.asr_model_name = "ASR_mandarin"
__C.model.asr_class_name = "ASR_Mandarin_Net"
__C.model.asr_num_classes = 408

# vad
__C.model.vad_mode = 3

# graph
__C.model.graph_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "impl", "rm_common_library/KeywordSearch/keyword_graph_formated.cmd")