import numpy as np
import os
from scipy.ndimage.morphology import binary_dilation
import struct
import torch
import webrtcvad

from impl.file_tools import load_module_from_disk
from impl.asr_feature_pyimpl import Feature
import impl.model_tool as model
import impl.asr_decode_pyimpl as Decode_Python
import impl.asr_decode_beamsearch as Decode_BeamSearch

from impl.rm_common_library.KeywordSearch.keyword_graph import PrimaryGraph, Graph
from impl.rm_common_library.KeywordSearch.command_graph import CommandGraph, Command
from impl.rm_common_library.KeywordSearch.token_pass_match import primary_token_pass


class KwsAsrApi():
    """
    KwsAsrApi
    """

    def __init__(self, bool_do_kws_wakeup=True, bool_do_asr=True, bool_gpu=True):
        self.bool_do_kws_wakeup = bool_do_kws_wakeup
        self.bool_do_asr = bool_do_asr
        self.bool_gpu = bool_gpu

        # cfg init
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RMAI_KWS_ASR_options_Canbin.py")
        self.cfg = load_module_from_disk(cfg_path).cfg
        # param_init
        self.param_init()

        # kws_asr_init
        self.kws_asr_init()

    def param_init(self):
        self.params_dict = {}

        # container
        self.params_dict['audio_data_container_np'] = np.array([])
        self.params_dict['feature_data_container_np'] = np.array([])
        self.params_dict['kws_container_np'] = np.array([])         # kws 结构容器中，用于滑窗输出结果
        self.params_dict['vad_bool_container'] = []                 # vad 结构容器中，用于判断连续 5s 输出
        self.params_dict['output_wave_list'] = []
        self.params_dict['asr_duplicate_counter'] = {}

        self.params_dict['asr_vad_audio_data_container_np'] = np.zeros(
            int(self.cfg.general.sample_rate * self.cfg.general.asr_vad_audio_data_ms / 1000.0))
        self.params_dict['asr_vad_flag'] = False
        self.params_dict['asr_vad_first_detect'] = True
        self.params_dict['asr_vad_activate_count'] = 0
        self.params_dict['asr_vad_activate_flag'] = False
        self.params_dict['asr_vad_activate_pos_id'] = 0
        self.params_dict['asr_vad_flag_count'] = 0
        self.params_dict['asr_vad_loop_times'] = 0

        self.params_dict['bool_wakeup'] = False
        self.params_dict['counter_asr'] = self.cfg.general.asr_suppression_counter - 1

    def kws_asr_init(self):
        # init model
        self.kws_init()
        self.asr_init()
        self.vad_init()
        self.graph_init()

    def run_kws_asr(self, audio_data):
        # init 
        asr_command=[]
        asr_user_input=''
        # 准备数据和特征
        self.papare_data_and_feature(audio_data.copy())

        # 如果语音特征未装满容器，不进行唤醒和关键词检测
        if self.params_dict['feature_data_container_np'].shape[0] < self.cfg.general.feature_container_time:
            return

        # asr_duplicate_update_counter，更新计数器，防止重复检测
        self.asr_duplicate_update_counter()

        # 方案一：进行 kws 唤醒词检测，若检测到唤醒词，进入控制词识别模式
        # kws
        if not self.params_dict['bool_wakeup']:
            bool_find_kws = self.run_kws()

            if bool_find_kws:
                # 打印结果
                print("\n===============!!!!!!!!!!!!!!===============")
                print("********************************************")
                print("** ")
                print("** [Information:] Device wakeup:", "wakeup")
                print("** ")
                print("********************************************\n")

                self.params_dict['bool_wakeup'] = True
                asr_user_input="小锐小锐"
                asr_command=Command()
                asr_command.tpe='A'
                asr_command.cmd = "Wakeup"
        else:
            
            bool_exit_wakeup, asr_output_tuple = self.run_asr_vad(audio_data)
            asr_user_input=asr_output_tuple[0]
            asr_command=asr_output_tuple[1]
            if bool_exit_wakeup:
                self.params_dict['bool_wakeup'] = False
                
                # 控制 asr 的间隔时间
                self.params_dict['counter_asr'] -= 1

            if len(asr_command):
                print("\n===============!!!!!!!!!!!!!!===============")
                print("********************************************")
                print("** ")
                print("** [Information:] Detect Command:", asr_command)
                print("** [Information:] User Input:", asr_user_input)
                print("** ")
                print("********************************************\n")
            
        # 方案二：进行 asr 检测，间隔一定时长
        # asr
        # 如果检测到唤醒词，则执行方案一
        if self.params_dict['bool_wakeup']:
            self.params_dict['counter_asr'] = 0
        else:
            self.params_dict['counter_asr'] += 1

        if self.params_dict['counter_asr'] == self.cfg.general.asr_suppression_counter:
            self.params_dict['counter_asr'] = 0

            asr_output_tuple = self.run_asr(False)
            asr_user_input=asr_output_tuple[0]
            asr_command=asr_output_tuple[1]
            # 打印结果
            # 检测是否为 小锐小锐_唤醒词
            '''
            if '小锐小锐_唤醒' in asr_output_string:
                self.params_dict['bool_wakeup'] = True
                asr_output_string = "wakeup "
            '''
            if len(asr_command):
                print("\n===============!!!!!!!!!!!!!!===============")
                print("********************************************")
                print("** ")
                print("** [Information:] Detect Command:", asr_command)
                print("** [Information:] User Input:", asr_user_input)
                print("** ")
                print("********************************************\n")
            else:
                print("\n** [Information:] Detecting ...\n")
        return (asr_command,asr_user_input)

    def run_vad(self, wav):
        # Compute the voice detection window size
        samples_per_window = (self.cfg.general.vad_window_length * self.cfg.general.sample_rate) // 1000

        # Trim the end of the audio to have a multiple of the window size
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]

        # Convert the float waveform to 16-bit mono PCM
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * self.cfg.general.int16_max)).astype(np.int16))

        # Perform voice activation detection
        voice_flags = []
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(self.vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                                  sample_rate=self.cfg.general.sample_rate))
        voice_flags = np.array(voice_flags)

        audio_mask = self.moving_average(voice_flags, self.cfg.general.vad_moving_average_width)
        audio_mask = np.round(audio_mask).astype(np.bool)

        # Dilate the voiced regions
        audio_mask = binary_dilation(audio_mask, np.ones(self.cfg.general.vad_max_silence_length + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)

        vad_bool = True if (audio_mask == True).sum() / len(audio_mask) > 0.0 else False
        self.params_dict['vad_bool_container'].append(vad_bool)

        if len(self.params_dict['vad_bool_container']) > self.cfg.general.vad_container_time:
            self.params_dict['vad_bool_container'] = self.params_dict['vad_bool_container'][
                                                     - self.cfg.general.vad_container_time:]

        assert len(self.params_dict['vad_bool_container']) <= self.cfg.general.vad_container_time
        # print("vad: {} {}".format(vad_bool, (audio_mask == True).sum()/len(audio_mask)))
        # print("vad_container: ", self.params_dict['vad_bool_container'])

        run_vad_bool = False
        if len(self.params_dict['vad_bool_container']) == self.cfg.general.vad_container_time:
            if np.array(self.params_dict['vad_bool_container']).sum() == 0:

                # 保证唤醒后，一定会将 3s 音频用于控制词识别
                if not self.params_dict['bool_wakeup']:
                    run_vad_bool = True

        return run_vad_bool

    # Smooth the voice detection with a moving average
    def moving_average(self, array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    def sample_rate(self):
        return self.cfg.general.sample_rate

    def window_size_samples(self):
        return self.cfg.general.window_size_samples

    def window_stride_samples(self):
        return self.cfg.general.window_stride_samples

    def papare_data_and_feature(self, audio_data):
        # audio data
        # 拼接语音数据
        if len(self.params_dict['audio_data_container_np']):
            audio_data = np.concatenate((self.params_dict['audio_data_container_np'], audio_data), axis=0)

        # 存储指定时间的音频，用于后续拼接语音特征
        self.params_dict['audio_data_container_np'] = audio_data[
                                                      len(audio_data) - self.cfg.general.window_container_samples:]

        # feature
        # 计算特征
        feature = Feature(self.cfg.general.sample_rate, int(self.cfg.general.feature_freq), int(self.cfg.general.nfilt))
        feature.get_mel_int_feature(audio_data)
        feature_data = feature.copy_mfsc_feature_int_to()

        # 拼接特征
        if not self.params_dict['feature_data_container_np'].shape[0]:
            self.params_dict['feature_data_container_np'] = feature_data
        elif self.params_dict['feature_data_container_np'].shape[0] < self.cfg.general.feature_container_time:
            self.params_dict['feature_data_container_np'] = np.concatenate((self.params_dict[
                                                                                'feature_data_container_np'][
                                                                            : -self.cfg.general.feature_remove_after_time],
                                                                            feature_data), axis=0)
        else:
            self.params_dict['feature_data_container_np'] = np.concatenate((self.params_dict[
                                                                                'feature_data_container_np'][
                                                                            self.cfg.general.feature_remove_before_time: -self.cfg.general.feature_remove_after_time],
                                                                            feature_data), axis=0)

    def kws_init(self):
        self.kws_net = None

        if not self.bool_do_kws_wakeup:
            return

        # init model
        if self.cfg.model.bool_caffe:
            self.kws_net = model.caffe_model_init(self.cfg.model.kws_prototxt_path,
                                                  self.cfg.model.kws_model_path,
                                                  self.cfg.model.kws_net_input_name,
                                                  self.cfg.model.kws_chw_params.split(","),
                                                  self.bool_gpu)
        elif self.cfg.model.bool_pytorch:
            self.kws_net = model.pytorch_kws_model_init(self.cfg.model.kws_chk_path,
                                                        self.cfg.model.kws_model_name,
                                                        self.cfg.model.kws_class_name,
                                                        self.cfg.model.kws_num_classes,
                                                        self.cfg.model.image_height,
                                                        self.cfg.model.image_weidth,
                                                        self.bool_gpu)
        else:
            raise Exception(
                "bool_caffe = {}, bool_pytorch = {}".format(self.cfg.model.bool_caffe, self.cfg.model.bool_pytorch))

    def asr_init(self):
        self.asr_net = None
        self.asr_decoder = None

        self.asr_beamsearch = None
        self.lm = None

        if not self.bool_do_asr:
            return 

        # init model
        if self.cfg.model.bool_caffe:
            self.asr_net = model.caffe_model_init(self.cfg.model.asr_prototxt_path,
                                                  self.cfg.model.asr_model_path,
                                                  self.cfg.model.asr_net_input_name,
                                                  self.cfg.model.asr_chw_params.split(","),
                                                  self.bool_gpu)
        elif self.cfg.model.bool_pytorch:
            self.asr_net = model.pytorch_asr_model_init(self.cfg.model.asr_chk_path,
                                                        self.cfg.model.asr_model_name,
                                                        self.cfg.model.asr_class_name,
                                                        self.cfg.model.asr_num_classes,
                                                        self.bool_gpu)
        else:
            raise Exception(
                "bool_caffe = {}, bool_pytorch = {}".format(self.cfg.model.bool_caffe, self.cfg.model.bool_pytorch))

        # init bpe dict 
        self.asr_decoder = Decode_Python.Decode()
        self.asr_decoder.init_symbol_list(self.cfg.model.asr_dict_path)
        
        # init lm
        if self.cfg.general.decode_id == 1:
            self.asr_decoder.init_lm_model(self.cfg.model.asr_lm_path)
            
            lexicon = Decode_BeamSearch.load_lexicon(self.cfg.model.asr_dict_path)
            self.asr_beamsearch = Decode_BeamSearch.BeamSearch(lexicon, beam_size=15)
            self.lm = Decode_BeamSearch.Ken_LM(self.cfg.model.asr_lm_path)


    def vad_init(self):
        self.vad = None
        self.vad = webrtcvad.Vad(mode=self.cfg.model.vad_mode)

    def graph_init(self):
        self.graph = None
        self.graph = CommandGraph.build(self.cfg.model.graph_path)
        #self.graph = Graph.build(self.cfg.model.graph_path)

    def run_kws(self):
        # init
        kws_score_list = []

        if not self.bool_do_kws_wakeup:
            return False

        # 滑窗，模型前传
        # 每次送入 1s 数据，只需要对 1s 音频特征进行滑窗，模型前传；否则，会出现重复检测
        kws_wakeup_times = int((self.cfg.general.feature_time) * 1.0 / self.cfg.general.kws_stride_feature_time) + 1

        # 对每次送入的 1s 数据进行模型前传
        for kws_wakeup_time in range(kws_wakeup_times):
            end_feature_time = self.params_dict['feature_data_container_np'].shape[0] - (
                    kws_wakeup_times - kws_wakeup_time) * self.cfg.general.kws_stride_feature_time
            start_feature_time = end_feature_time - int(self.cfg.general.kws_feature_time)
            assert start_feature_time >= 0, "kws wakeup model 特征时间维度太大， 处理音频数据无法获得 {} 次滑窗结果".format(kws_wakeup_times)

            feature_data_kws = self.params_dict['feature_data_container_np'][start_feature_time: end_feature_time, :]
            feature_data_kws = feature_data_kws.astype(np.float32)

            if self.cfg.model.bool_caffe:
                net_output = model.caffe_model_forward(self.kws_net,
                                                       feature_data_kws,
                                                       self.cfg.model.kws_net_input_name,
                                                       self.cfg.model.kws_net_output_name,
                                                       self.cfg.model.kws_transpose)
            elif self.cfg.model.bool_pytorch:
                net_output = model.pytorch_model_forward(self.kws_net,
                                                         feature_data_kws,
                                                         self.bool_gpu)

            net_output = np.squeeze(net_output)
            kws_score_list.append(net_output.copy())

        # 如果有保留的 kws 结果，进行拼接
        kws_score_np = np.array(kws_score_list)
        if len(self.params_dict['kws_container_np']):
            kws_score_np = np.concatenate((self.params_dict['kws_container_np'], kws_score_np), axis=0)

        bool_find_kws = False
        for kws_idx in range(len(kws_score_np) + 1 - kws_wakeup_times):
            # 滑窗，获得后处理结果
            detected_number = 0
            for kws_times in range(kws_wakeup_times):
                if kws_score_np[kws_idx + kws_times][-1] > self.cfg.general.kws_detection_threshold:
                    detected_number += 1

            if detected_number >= kws_wakeup_times * self.cfg.general.kws_detection_number_threshold:
                bool_find_kws = True

        if bool_find_kws:
            self.params_dict['kws_container_np'] = np.zeros(np.array(kws_score_list).shape)
        else:
            # 存储一定时间的 kws 结果，用于后续滑窗获得结果
            self.params_dict['kws_container_np'] = np.array(kws_score_list)

        return bool_find_kws

    def run_asr(self, contorl_kws_bool=True):
        if not self.bool_do_asr:
            return ('',[])

        asr_tuple = self.run_asr_normal(contorl_kws_bool)
        user_string=asr_tuple[0]
        asr_string=asr_tuple[1]
        if len(asr_string):
            asr_string = self.asr_duplicate_check(asr_string)
        return (user_string,asr_string)

    def run_asr_normal(self, contorl_kws_bool=True):
        canbin_mode=True
        result_tuple=('',[])
        if not self.bool_do_asr:
            return result_tuple

        # 获取特征
        feature_data_asr = self.params_dict['feature_data_container_np'][-self.cfg.general.asr_feature_time:, :].astype(np.float32)
        
        # 模型前向传播
        if self.cfg.model.bool_caffe:
            net_output = model.caffe_model_forward(self.asr_net,
                                                   feature_data_asr,
                                                   self.cfg.model.asr_net_input_name,
                                                   self.cfg.model.asr_net_output_name)
            net_output = np.squeeze(net_output)
            net_output = net_output.T
        elif self.cfg.model.bool_pytorch:
            net_output = model.pytorch_model_forward(self.asr_net,
                                                     feature_data_asr,
                                                     self.bool_gpu)
            net_output = np.squeeze(net_output)

        # decode
        if self.cfg.general.decode_id == 0:
            self.asr_decoder.ctc_decoder(net_output)
        elif self.cfg.general.decode_id == 1:
            self.asr_decoder.beamsearch_decoder(net_output, 5, 0, bswt=1.0, lmwt=0.3)
        else:
            raise Exception("[Unknow:] cfg.general.decode_id = {}".format(self.cfg.general.decode_id))

        if self.cfg.general.language_id == 0:
            if(canbin_mode):
                symbol_list = self.asr_decoder.output_symbol_list()
                if(len(symbol_list)==0):
                    return ('',[])
                if(0):
                    detect_token = primary_token_pass(symbol_list, self.graph)
                    if not detect_token is None:
                        result_tuple = Decode_Python.get_ouststr(detect_token)
                        if('clamdown' in result_tuple[0] or 'hidden' in result_tuple[0]):
                            return result_tuple
                        else:
                            return ('',[])
                else:
                    detect_token = self.graph.parse_command(symbol_list)
                    if(len(detect_token.commands)==0 or (detect_token.commands[0].tpe!='F' and detect_token.commands[0].tpe!='H')):
                        return ('',[])
                    else:
                        return (detect_token.symbols,detect_token.commands)
            else:
                self.asr_decoder.show_result_id()
                self.asr_decoder.show_symbol()
                # result_string = self.asr_decoder.output_symbol()
                self.asr_decoder.match_keywords_chinese(self.cfg.general.kws_list, self.cfg.general.kws_dict)
                result_string = self.asr_decoder.output_control_result_string(self.cfg.general.control_kws_list,
                                                                            contorl_kws_bool)
        elif self.cfg.general.language_id == 1:
            pass
        else:
            raise Exception("[Unknow:] cfg.general.language_id = {}".format(self.cfg.general.language_id))

        return result_tuple

    def run_asr_vad(self, audio_data):
        asr_vad_max_activate_count=16
        asr_vad_activate_threshold=10
        asr_vad_deactivate_threshold=2
        result_command=Command()
        # init 
        result_tuple = ('',[])
        self.params_dict['asr_vad_audio_data_container_np'][self.cfg.general.sample_rate * (int(self.cfg.general.asr_vad_audio_data_ms / 1000.0)- 1):] = audio_data

        # 进入 asr vad 模式
        if not self.params_dict['asr_vad_flag']:
            self.params_dict['asr_vad_flag'] = True

        # 状态变化
        self.params_dict['asr_vad_flag_count'] += 1
        print("[Information:] Go into run_asr_vad, asr_vad_flag_count: ", self.params_dict['asr_vad_flag_count'])

        # 循环检测 vad
        audio_data_len = len(audio_data)
        for i in range(0, audio_data_len - 480, 480):

            # vad 计数，表示当前时刻是否已经结束
            vad_wav = audio_data[i: i + 480].astype(np.int16).tobytes()
            if self.vad.is_speech(vad_wav, sample_rate=self.cfg.general.sample_rate):
                self.params_dict['asr_vad_activate_count'] += 1
            else:
                self.params_dict['asr_vad_activate_count'] -= 1
            self.params_dict['asr_vad_activate_count'] = max(0, min(self.params_dict['asr_vad_activate_count'], asr_vad_max_activate_count))

            # vad 起点位置确立
            if (self.params_dict['asr_vad_activate_count'] > asr_vad_activate_threshold and self.params_dict[
                'asr_vad_activate_flag'] == False):
                self.params_dict['asr_vad_activate_flag'] = True
                self.params_dict['asr_vad_activate_pos_id'] = self.cfg.general.sample_rate * 5 + i - asr_vad_activate_threshold * 480

            # vad 终点位置确立
            if self.params_dict['asr_vad_activate_count'] < asr_vad_deactivate_threshold:

                # 判断是否确立起始位置
                if not self.params_dict['asr_vad_activate_flag'] == True:
                    continue
                
                # 方式一：收集音频，重新计算特征
                # # 获取音频数据
                # wave = self.params_dict['asr_vad_audio_data_container_np'][
                #          self.params_dict['asr_vad_activate_pos_id']: min(self.cfg.general.sample_rate * 5 + i + 960,
                #                                                     self.cfg.general.sample_rate * 6)]

                # # 计算有效音频长度，判断是否小于检测最短长度
                # if len(wave) < int(self.cfg.general.sample_rate * self.cfg.general.asr_vad_minitime_threshold_s[self.params_dict['asr_vad_loop_times']]):
                #     continue

                # # 识别结果
                # print("[Information:] 正常检测到 vad 起止位置，识别返回结果")
                # result_tuple = self.asr_vad_detecte(wave)

                # 方式二：直接从容器里抽取特征
                # 获取音频数据
                start_pos_id = self.params_dict['asr_vad_activate_pos_id']
                end_pos_id =  min(self.cfg.general.sample_rate * 5 + i + 960, self.cfg.general.sample_rate * 6)
                wave_len = end_pos_id - start_pos_id

                # 计算有效音频长度，判断是否小于检测最短长度
                if wave_len < int(self.cfg.general.sample_rate * self.cfg.general.asr_vad_minitime_threshold_s[self.params_dict['asr_vad_loop_times']]):
                    continue

                # 识别结果
                print("[Information:] 正常检测到 vad 起止位置，识别返回结果")
                result_tuple = self.asr_vad_detecte_pos(start_pos_id, end_pos_id)

                if len(result_tuple[1]) != 0:
                    self.params_dict['asr_vad_first_detect']=False
                    # 清空
                    self.asr_vad_state_reset(reset_vad_falg=False)
                    return False, result_tuple
                else:
                    # 状态变化
                    result_command.tpe='W'
                    self.params_dict['asr_vad_loop_times'] += 1
                    if self.params_dict['asr_vad_first_detect']==False or self.params_dict['asr_vad_loop_times'] >= self.cfg.general.asr_vad_loop_threshold:
                        # 清空
                        self.asr_vad_state_reset()
                        if(self.params_dict['asr_vad_first_detect']==False):
                            result_command.cmd=self.cfg.general.string_end_safety
                            return True, ('',[result_command])
                        else:
                            result_command.cmd=self.cfg.general.string_end_loop
                            return True, ('',[result_command])
                    else:
                        self.params_dict['asr_vad_first_detect']=False
                        self.asr_vad_state_reset(reset_vad_falg=False)
                        result_command.cmd=self.cfg.general.string_warring
                        return False, ('',[result_command])

        # 缓存，移位
        self.params_dict['asr_vad_audio_data_container_np'][:self.cfg.general.sample_rate * 5] = self.params_dict[
                                                                                                     'asr_vad_audio_data_container_np'][
                                                                                                 -self.cfg.general.sample_rate * 5:]

        if self.params_dict['asr_vad_activate_pos_id'] > 0:
            self.params_dict['asr_vad_activate_pos_id'] = max(0, self.params_dict['asr_vad_activate_pos_id'] - self.cfg.general.sample_rate)

        # check：判断没有说话
        if (self.params_dict['asr_vad_flag_count'] >= self.cfg.general.asr_vad_silence_threshold_s
                and self.params_dict['asr_vad_activate_flag'] == False):
            print("[Information:] silence >= {}，未检测到声音".format(self.cfg.general.asr_vad_silence_threshold_s))
            # 清空
            self.asr_vad_state_reset()
            result_command.tpe='W'
            result_command.cmd=self.cfg.general.string_end_safety
            if(self.params_dict['asr_vad_first_detect']==False):
                return True, ('',[result_command])
            else:
                result_command.cmd=self.cfg.general.string_silence_warring
                return True, ('',[result_command])

        # check：判断超过时长
        if self.params_dict['asr_vad_flag_count'] >= self.cfg.general.asr_vad_overtime_threshold_s:
            print("[Information:] overtime >= {}，超长时间检测".format(self.cfg.general.asr_vad_overtime_threshold_s))

            # 方式一：收集音频，重新计算特征
            # # 音频数据
            # wave = self.params_dict['asr_vad_audio_data_container_np'][
            #             self.params_dict['asr_vad_activate_pos_id']: ]

            # # 识别结果
            # result_tuple = self.asr_vad_detecte(wave)

            # 方式二：直接从容器里抽取特征
            # 获取音频数据
            start_pos_id = self.params_dict['asr_vad_activate_pos_id']
            end_pos_id = self.cfg.general.sample_rate

            # 识别结果
            print("[Information:] 正常检测到 vad 起止位置，识别返回结果")
            result_tuple = self.asr_vad_detecte_pos(start_pos_id, end_pos_id)

            if len(result_tuple[1]) != 0:
                # 清空
                self.params_dict['asr_vad_first_detect']=False
                self.asr_vad_state_reset(reset_vad_falg=False)
                return False, result_tuple
            else:
                # 状态变化
                result_command.tpe='W'
                self.params_dict['asr_vad_loop_times'] += 1
                if self.params_dict['asr_vad_first_detect']==False or self.params_dict['asr_vad_loop_times'] >= self.cfg.general.asr_vad_loop_threshold:
                    # 清空
                    self.asr_vad_state_reset()
                    if(self.params_dict['asr_vad_first_detect']==False):
                        result_command.cmd=self.cfg.general.string_end_safety
                        return True, ('',[result_command])
                    else:
                        result_command.cmd=self.cfg.general.string_end_loop
                        return True, ('',[result_command])
                else:
                    self.params_dict['asr_vad_first_detect']=False
                    self.asr_vad_state_reset(reset_vad_falg=False)
                    result_command.cmd=self.cfg.general.string_warring
                    return False, ('',[result_command])

        # 返回，输出
        if self.params_dict['asr_vad_flag']:
            return False, result_tuple
        else:
            # 清空
            self.asr_vad_state_reset(reset_vad_falg=False)
            return False, result_tuple

    def asr_vad_detecte(self, wave):
        # init
        result = ('',[])

        # feature
        # 计算特征
        feature = Feature(self.cfg.general.sample_rate, int(self.cfg.general.feature_freq),
                            int(self.cfg.general.nfilt))
        feature.get_mel_int_feature(wave)
        feature_data = feature.copy_mfsc_feature_int_to().astype(np.float32)

        # 模型前向传播
        if self.cfg.model.bool_caffe:
            net_output = model.caffe_model_forward(self.asr_net,
                                                    feature_data,
                                                    self.cfg.model.asr_net_input_name,
                                                    self.cfg.model.asr_net_output_name)
            net_output = np.squeeze(net_output)
            net_output = net_output.T
        elif self.cfg.model.bool_pytorch:
            net_output = model.pytorch_model_forward(self.asr_net,
                                                        feature_data,
                                                        self.bool_gpu)
            net_output = np.squeeze(net_output)

        # decode
        if self.cfg.general.decode_id == 0:
            raise NotImplementedError
        elif self.cfg.general.decode_id == 1:
            net_output = torch.from_numpy(net_output)
            # symbol_list = self.asr_beamsearch.prefix_beam_search(net_output, lm=self.lm)
            symbol_list = self.asr_beamsearch.prefix_beam_search_contextbias(net_output, lm=self.lm, lm_weight=0.3)
        else:
            raise Exception("[Unknow:] cfg.general.decode_id = {}".format(self.cfg.general.decode_id))

        if self.cfg.general.language_id == 0:
            #print('beamsearch result: ',symbol_list)
            if(0):
                detect_token = primary_token_pass(symbol_list, self.graph)
                result = Decode_Python.get_ouststr(detect_token)
            else:
                result = self.graph.parse_command(symbol_list)
                result = (result.symbols,result.commands)

        elif self.cfg.general.language_id == 1:
            raise NotImplementedError
        else:
            raise Exception("[Unknow:] cfg.general.language_id = {}".format(self.cfg.general.language_id))

        return result

    def asr_vad_detecte_pos(self, start_pos_id, end_pos_id):
        # init
        result = ('',[])

        # feature
        # 获取特征
        start_pos_time = max(int(start_pos_id / self.cfg.general.sample_rate * 100), 0)
        end_pos_time = min(int(end_pos_id / self.cfg.general.sample_rate * 100), self.params_dict['feature_data_container_np'].row())
        feature_data_asr = self.params_dict['feature_data_container_np'][start_pos_time : end_pos_time, :].astype(np.float32)

        # 模型前向传播
        if self.cfg.model.bool_caffe:
            net_output = model.caffe_model_forward(self.asr_net,
                                                    feature_data_asr,
                                                    self.cfg.model.asr_net_input_name,
                                                    self.cfg.model.asr_net_output_name)
            net_output = np.squeeze(net_output)
            net_output = net_output.T
        elif self.cfg.model.bool_pytorch:
            net_output = model.pytorch_model_forward(self.asr_net,
                                                        feature_data_asr,
                                                        self.bool_gpu)
            net_output = np.squeeze(net_output)

        # decode
        if self.cfg.general.decode_id == 0:
            raise NotImplementedError
        elif self.cfg.general.decode_id == 1:
            net_output = torch.from_numpy(net_output)
            # symbol_list = self.asr_beamsearch.prefix_beam_search(net_output, lm=self.lm)
            symbol_list = self.asr_beamsearch.prefix_beam_search_contextbias(net_output, lm=self.lm, lm_weight=0.3)
        else:
            raise Exception("[Unknow:] cfg.general.decode_id = {}".format(self.cfg.general.decode_id))

        if self.cfg.general.language_id == 0:
            #print('beamsearch result: ',symbol_list)
            if(0):
                detect_token = primary_token_pass(symbol_list, self.graph)
                result = Decode_Python.get_ouststr(detect_token)
            else:
                result = self.graph.parse_command(symbol_list)
                result = (result.symbols,result.commands)

        elif self.cfg.general.language_id == 1:
            raise NotImplementedError
        else:
            raise Exception("[Unknow:] cfg.general.language_id = {}".format(self.cfg.general.language_id))

        return result

    def asr_vad_state_reset(self, reset_vad_falg = True):
        self.params_dict['asr_vad_audio_data_container_np'] = np.zeros(
            int(self.cfg.general.sample_rate * self.cfg.general.asr_vad_audio_data_ms / 1000.0))
        self.params_dict['asr_vad_activate_pos_id'] = 0
        self.params_dict['asr_vad_flag_count'] = 0
        self.params_dict['asr_vad_activate_flag'] = False
        if reset_vad_falg:
            self.params_dict['asr_vad_flag'] = False
            self.params_dict['asr_vad_loop_times'] = 0
            self.params_dict['asr_vad_first_detect']=True

    def asr_duplicate_update_counter(self):
        for key in self.params_dict['asr_duplicate_counter']:
            if self.params_dict['asr_duplicate_counter'][key] > 0:
                self.params_dict['asr_duplicate_counter'][key] = self.params_dict['asr_duplicate_counter'][
                                                                     key] - self.cfg.general.window_size_ms
                #print(key, self.params_dict['asr_duplicate_counter'][key])

    def asr_duplicate_check(self, asr_string):
        if(1):
            res_list=[]
            for idx in range(len(asr_string)):
                if 'Wakeup' in asr_string[idx].cmd:
                    res_list.append(asr_string[idx])
                    continue
                if asr_string[idx].cmd not in self.params_dict['asr_duplicate_counter']:
                    self.params_dict['asr_duplicate_counter'][asr_string[idx].cmd] = self.cfg.general.total_time_ms
                    res_list.append(asr_string[idx])
                else:
                    if self.params_dict['asr_duplicate_counter'][asr_string[idx].cmd] > 0:
                        continue
                    else:
                        self.params_dict['asr_duplicate_counter'][asr_string[idx].cmd] = self.cfg.general.total_time_ms
                        res_list.append(asr_string[idx])
            return res_list
        else:
            res_string = ""
            tmp_string = asr_string.split(' ')
            for idx in range(len(tmp_string)):
                if '小锐小锐_唤醒' in tmp_string[idx]:
                    res_string += tmp_string[idx] + " "
                    continue
                if tmp_string[idx] not in self.params_dict['asr_duplicate_counter']:
                    self.params_dict['asr_duplicate_counter'][tmp_string[idx]] = self.cfg.general.total_time_ms
                    res_string += tmp_string[idx] + " "
                else:
                    if self.params_dict['asr_duplicate_counter'][tmp_string[idx]] > 0:
                        continue
                    else:
                        self.params_dict['asr_duplicate_counter'][tmp_string[idx]] = self.cfg.general.total_time_ms
                        res_string += tmp_string[idx] + " "
            return res_string