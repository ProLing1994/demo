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

# container
__C.general.audio_container_ms = 100                # 语音数据容器中，装有音频数据 100 ms
__C.general.audio_container_time = 10               # 语音数据容器中，装有音频数据 100 ms，对应特征维度 10
__C.general.feature_container_time = 296            # 语音特征容器中，装有时间维度 296
__C.general.feature_remove_after_time = 6           # 为保证特征一致，拼接特征需要丢弃最后的时间维度 6
__C.general.feature_remove_before_time = 100        # 为保证特征一致，拼接特征需要丢弃之前的时间维度 100
__C.general.vad_container_time = 4                  # vad 容器，判断连续 4s 中是否全部为静音，用于停止后续操作

# chinese key words
__C.general.kws_list = ['小锐小锐_唤醒', 
                        '他妈的_脏话', '傻逼_脏话', '我操_脏话', '脑残_脏话', '妈了个逼_脏话', 
                        '去不了_拒载', '带不了你_拒载', '走不了_拒载', '不去不去_拒载', '不顺路_拒载',
                        '要加钱_议价', '打表就不去_议价', '打表不行_议价', 
                        '弄死你_威胁', '要是敢报警_威胁', '手机拿出来_威胁', '自己解锁还是我帮你_威胁', '要钱还是要命_威胁', '把钱拿出来_威胁',
                        '打开地图_控制', '关闭地图_控制', '退出地图_控制', '打开蓝牙_控制', '关闭蓝牙_控制', '退出蓝牙_控制', '打开收音机_控制', '关闭收音机_控制', '退出收音机_控制',
                        '上一频道_控制', '下一频道_控制', '音量加大_控制', '音量减小_控制', '导航到_控制', '深圳北站_控制', '竹子林地铁站_控制', '机场_控制', '世界之窗_控制', '查看营收_控制',
                        '高德电召_控制', '呼叫乘客_控制', '上个频道_控制', '下个频道_控制', '打电话_控制', '暂停_控制', '继续_控制', '电召_控制', '回拨_控制', '接单_控制', '抢单_控制']
__C.general.kws_dict = {'小锐小锐_唤醒':'xiao rui xiao rui',
                        
                        '他妈的_脏话':'ta ma de',
                        '傻逼_脏话':'sha bi',
                        '我操_脏话':'wo cao', 
                        '脑残_脏话':'nao can',
                        '妈了个逼_脏话':'ma le ge bi', 

                        '去不了_拒载':'qu bu liao', 
                        '带不了你_拒载':'dai bu liao ni', 
                        '走不了_拒载':'zou bu liao', 
                        '不去不去_拒载':'bu qu bu qu', 
                        '不顺路_拒载':'bu shun lu',

                        '要加钱_议价':'yao jia qian', 
                        '打表就不去_议价':'da biao jiu bu qu',
                        '打表不行_议价':'da biao bu xing', 

                        '弄死你_威胁':'nong shi ni', 
                        '要是敢报警_威胁':'yao shi gan bao jing', 
                        '手机拿出来_威胁':'shou ji na chu lai', 
                        '自己解锁还是我帮你_威胁':'zi ji jie suo hai shi wo bang ni', 
                        '要钱还是要命_威胁':'yao qian hai shi yao ming',
                        '把钱拿出来_威胁':'ba qian na chu lai',

                        '打开地图_控制':'da kai di tu', 
                        '关闭地图_控制':'guan bi di tu', 
                        '退出地图_控制':'tui chu di tu', 
                        '打开蓝牙_控制':'da kai lan ya', 
                        '关闭蓝牙_控制':'guan bi lan ya',
                        '退出蓝牙_控制':'tui chu lan ya',
                        '打开收音机_控制':'da kai shou yin ji', 
                        '关闭收音机_控制':'guan bi shou yin ji', 
                        '退出收音机_控制':'tui chu shou yin ji', 
                        '上一频道_控制':'shang yi pin dao', 
                        '下一频道_控制':'xia yi pin dao',
                        '音量加大_控制':'yin liang jia da',
                        '音量减小_控制':'yin liang jian xiao', 
                        '导航到_控制':'dao hang dao', 
                        '深圳北站_控制':'shen zhen bei zan', 
                        '竹子林地铁站_控制':'zhu zi lin di tie zan', 
                        '机场_控制':'ji chang',
                        '世界之窗_控制':'shi jie zi chuang',
                        '查看营收_控制':'cha kan yin shou', 
                        '高德电召_控制':'gao de dian zhao', 
                        '呼叫乘客_控制':'hu jiao cheng ke', 
                        '上个频道_控制':'shang ge pin dao', 
                        '下个频道_控制':'xia ge pin dao',
                        '打电话_控制':'da dian hua',
                        '暂停_控制':'zan ting',
                        '继续_控制':'ji xu', 
                        '电召_控制':'dian zhao', 
                        '回拨_控制':'hui bo', 
                        '接单_控制':'jie dan', 
                        '抢单_控制':'qiang dan',
                        }
__C.general.control_kws_list = ['打开地图_控制', '关闭地图_控制', '退出地图_控制', '打开蓝牙_控制', '关闭蓝牙_控制', '退出蓝牙_控制', '打开收音机_控制', '关闭收音机_控制', '退出收音机_控制',
                                '上一频道_控制', '下一频道_控制', '音量加大_控制', '音量减小_控制', '导航到_控制', '深圳北站_控制', '竹子林地铁站_控制', '机场_控制', '世界之窗_控制', '查看营收_控制',
                                '高德电召_控制', '呼叫乘客_控制', '上个频道_控制', '下个频道_控制', '打电话_控制', '暂停_控制', '继续_控制', '电召_控制', '回拨_控制', '接单_控制', '抢单_控制']

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
__C.model.asr_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "asr_mandarin_taxi_16k_64.caffemodel")
__C.model.asr_prototxt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "asr_mandarin_taxi_16k_64_396.prototxt")
__C.model.asr_dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "asr_mandarin_dict_taxi.txt")
__C.model.asr_lm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "3gram_asr_mandarin_taxi_408.bin")
__C.model.asr_net_input_name = "data"
__C.model.asr_net_output_name = "prob"
__C.model.asr_chw_params = "1,296,64"

## pytorch 
__C.model.asr_chk_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "taxi_16k_64dim.pth")
__C.model.asr_model_name = "ASR_mandarin"
__C.model.asr_class_name = "ASR_Mandarin_Net"
__C.model.asr_num_classes = 408

# vad
__C.model.vad_mode = 3