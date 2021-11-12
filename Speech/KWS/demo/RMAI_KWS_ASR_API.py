from datetime import datetime
import numpy as np
import os
import sys

import impl.model_tool as model

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
import ASR.impl.asr_data_loader_pyimpl as WaveLoader_Python
from ASR.impl.asr_feature_pyimpl import Feature
import ASR.impl.asr_decode_pyimpl as Decode_Python
import SV.demo.impl.model_tool as sv_model

sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.file_tools import load_module_from_disk


class KwsAsrApi():
    """
    KwsAsrApi
    """
    def __init__(self, cfg_path, bool_do_kws_weakup=True, bool_do_asr=True, bool_do_sv=False, bool_gpu=True):
        self.bool_do_kws_weakup = bool_do_kws_weakup
        self.bool_do_asr = bool_do_asr
        self.bool_do_sv = bool_do_sv
        self.bool_gpu = bool_gpu

        # cfg init
        self.cfg = load_module_from_disk(cfg_path).cfg
        
        # param_init
        self.param_init()

        # kws_asr_init
        self.kws_asr_init()

    def param_init(self):
        self.params_dict = {}
        self.output_dict = {}

        # container
        self.params_dict['audio_data_container_np'] = np.array([])
        self.params_dict['feature_data_container_np'] = np.array([])
        self.params_dict['kws_container_np'] = np.array([])         # kws 结构容器中，用于滑窗输出结果
        self.params_dict['sv_embedding_container'] = []                       # sv 结构容器中，用于存储 embedding
        self.params_dict['asr_duplicate_counter'] = {}

        self.params_dict['bool_weakup'] = False
        self.params_dict['counter_weakup'] = 0
        self.params_dict['counter_asr'] = self.cfg.general.asr_suppression_counter - 1
        
        self.output_dict['output_data_list'] = []
        self.output_dict['output_kws_id'] = 1

        # mkdir
        if self.cfg.general.bool_output_wave:
            if not os.path.exists(self.cfg.test.output_folder):
                os.makedirs(self.cfg.test.output_folder)

    def kws_asr_init(self):
        # init model
        self.kws_init()
        self.asr_init_normal()
        self.sv_init()

        if self.cfg.general.asr_second_on:
            self.asr_init_second()

    def run_kws_asr(self, audio_data):
        # init 
        output_str = ''

        # 准备数据和特征
        self.papare_data_and_feature(audio_data)

        # 如果语音特征未装满容器，不进行唤醒和关键词检测
        if self.params_dict['feature_data_container_np'].shape[0] < self.cfg.general.feature_container_time:
            return

        # asr_duplicate_update_counter，更新计数器，防止重复检测
        self.asr_duplicate_update_counter()

        # 方案一：进行 kws 唤醒词检测，若检测到唤醒词，未来三秒进行 asr 检测
        # kws
        if not self.params_dict['bool_weakup']:
            bool_find_kws = self.run_kws()

            if bool_find_kws:
                # 打印结果
                print("\n===============!!!!!!!!!!!!!!===============")
                print("********************************************")
                print("** ")
                print("** [Information:] Device Weakup:", "Weakup")
                print("** ")
                print("********************************************\n")

                self.params_dict['bool_weakup'] = True
                output_str += "Weakup "

                # sv
                self.run_sv()

                # save audio
                self.output_wave("Weakup")
        else:
            self.params_dict['counter_weakup'] += 1
            if self.params_dict['counter_weakup'] >= self.cfg.general.kws_suppression_counter:
                self.params_dict['counter_weakup'] = 0
                self.params_dict['bool_weakup'] = False
                # 控制 asr 的间隔时间
                self.params_dict['counter_asr'] -= 1

                # asr
                asr_output_string = self.run_asr(True)
            
                # 打印结果
                if len(asr_output_string):
                    print("\n===============!!!!!!!!!!!!!!===============")
                    print("********************************************")
                    print("** ")
                    print("** [Information:] Detect Command:", asr_output_string)
                    print("** ")
                    print("********************************************\n")
                    output_str += asr_output_string + ' '

                    # save audio
                    self.output_wave("ASR_" + asr_output_string)
                else:
                    print("\n** [Information:] Detecting ...\n")

        # 方案二：进行 asr 检测，间隔一定时长
        # asr
        # 如果检测到唤醒词，则执行方案一
        if self.params_dict['bool_weakup']:
            self.params_dict['counter_asr'] = 0
        else:
            self.params_dict['counter_asr'] += 1

        if self.params_dict['counter_asr'] == self.cfg.general.asr_suppression_counter:
            self.params_dict['counter_asr'] = 0

            asr_output_string = self.run_asr(False)

            # 打印结果
            if len(asr_output_string):
                print("\n===============!!!!!!!!!!!!!!===============")
                print("********************************************")
                print("** ")
                print("** [Information:] Detect Command:", asr_output_string)
                print("** ")
                print("********************************************\n")
                output_str += asr_output_string + ' '

                # save audio
                self.output_wave("ASR_" + asr_output_string)
            else:
                print("\n** [Information:] Detecting ...\n")
        return output_str

    def input_wav(self):
        return self.cfg.test.input_wav

    def sample_rate(self):
        return self.cfg.general.sample_rate

    def window_size_samples(self):
        return self.cfg.general.window_size_samples

    def window_stride_samples(self):
        return self.cfg.general.window_stride_samples

    def papare_data_and_feature(self, audio_data):

        # 加载音频数据，用于打印输出
        if self.cfg.general.bool_output_wave:
            if len(self.output_dict['output_data_list']) < self.cfg.general.total_time_samples:
                self.output_dict['output_data_list'].extend(audio_data)
            else:
                self.output_dict['output_data_list'] = self.output_dict['output_data_list'][self.cfg.general.window_size_samples:]
                self.output_dict['output_data_list'].extend(audio_data)

        # audio data
        # 拼接语音数据
        if len(self.params_dict['audio_data_container_np']):
            audio_data = np.concatenate((self.params_dict['audio_data_container_np'], audio_data), axis=0)
        
        # 存储指定时间的音频，用于后续拼接语音特征
        self.params_dict['audio_data_container_np'] = audio_data[len(audio_data) - self.cfg.general.window_container_samples:]

        # feature
        # 计算特征
        feature = Feature(self.cfg.general.sample_rate, int(self.cfg.general.feature_freq), int(self.cfg.general.nfilt))
        feature.get_mel_int_feature(audio_data)
        feature_data = feature.copy_mfsc_feature_int_to()

        # 拼接特征
        if not self.params_dict['feature_data_container_np'].shape[0]:
            self.params_dict['feature_data_container_np'] = feature_data
        elif self.params_dict['feature_data_container_np'].shape[0] < self.cfg.general.feature_container_time:
            self.params_dict['feature_data_container_np'] = np.concatenate((self.params_dict['feature_data_container_np'][: -self.cfg.general.feature_remove_after_time], feature_data), axis=0)
        else:
            self.params_dict['feature_data_container_np'] = np.concatenate((self.params_dict['feature_data_container_np'][self.cfg.general.feature_remove_before_time: -self.cfg.general.feature_remove_after_time], feature_data), axis=0)

    def kws_init(self):
        self.kws_net = None

        if not self.bool_do_kws_weakup:
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
            raise Exception("bool_caffe = {}, bool_pytorch = {}".format(self.cfg.model.bool_caffe, self.cfg.model.bool_pytorch))

    def asr_init_normal(self):
        self.asr_net = None
        self.asr_decoder = None

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
            raise Exception("bool_caffe = {}, bool_pytorch = {}".format(self.cfg.model.bool_caffe, self.cfg.model.bool_pytorch))

        # init bpe dict 
        self.asr_decoder = Decode_Python.Decode()
        self.asr_decoder.init_symbol_list(self.cfg.model.asr_dict_path)

        # init lm
        if self.cfg.general.decode_id == 1:
            self.asr_decoder.init_lm_model(self.cfg.model.asr_lm_path)

    def asr_init_second(self):
        self.asr_net_second = None
        self.asr_decoder_second = None

        if not self.bool_do_asr:
            return 

        # init model
        # init phoneme net
        if self.cfg.model.bool_caffe:
            self.asr_net_second = model.caffe_model_init(self.cfg.model.asr_second_prototxt_path,
                                                            self.cfg.model.asr_second_model_path, 
                                                            self.cfg.model.asr_second_net_input_name, 
                                                            self.cfg.model.asr_second_chw_params.split(","), 
                                                            self.bool_gpu)
        else:
            # TO DO，未实现
            raise Exception("bool_caffe = {}, bool_pytorch = {}".format(self.cfg.model.bool_caffe, self.cfg.model.bool_pytorch))

        # init phoneme dict 
        self.asr_decoder_second = Decode_Python.Decode()
        self.asr_decoder_second.init_symbol_list(self.cfg.model.asr_second_dict_path)
 
        # init lm
        if self.cfg.general.second_decode_id == 1:
            self.asr_decoder_second.init_lm_model(self.cfg.model.asr_second_lm_path)

    def sv_init(self):
        self.sv_net = None

        if not self.bool_do_sv:
            return 

        # cfg init
        self.cfg_sv = load_module_from_disk(self.cfg.model.sv_config_file).cfg

        # init model
        if self.cfg.model.bool_caffe:
            pass
        elif self.cfg.model.bool_pytorch:
            self.sv_net = sv_model.pytorch_sv_model_init(self.cfg_sv,
                                                        self.cfg.model.sv_chk_path, 
                                                        self.cfg.model.sv_model_name, 
                                                        self.cfg.model.sv_class_name, 
                                                        self.bool_gpu)
        else:
            raise Exception("bool_caffe = {}, bool_pytorch = {}".format(self.cfg.model.bool_caffe, self.cfg.model.bool_pytorch))
            
    def run_kws(self):
        # init
        kws_score_list = []
        
        if not self.bool_do_kws_weakup:
            return False

        # 滑窗，模型前传
        # 每次送入 1s 数据，只需要对 1s 音频特征进行滑窗，模型前传；否则，会出现重复检测
        kws_weakup_times =  int((self.cfg.general.feature_time) * 1.0 / self.cfg.general.kws_stride_feature_time) + 1

        # 对每次送入的 1s 数据进行模型前传
        for kws_weakup_time in range(kws_weakup_times):
            end_feature_time = self.params_dict['feature_data_container_np'].shape[0] - (kws_weakup_times - kws_weakup_time) * self.cfg.general.kws_stride_feature_time
            start_feature_time = end_feature_time - int(self.cfg.general.kws_feature_time)
            assert start_feature_time >= 0, "kws weakup model 特征时间维度太大， 处理音频数据无法获得 {} 次滑窗结果".format(kws_weakup_times)

            feature_data_kws = self.params_dict['feature_data_container_np'][start_feature_time: end_feature_time,:]
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
        for kws_idx in range(len(kws_score_np) + 1 - kws_weakup_times):
            # 滑窗，获得后处理结果
            detected_number = 0 
            for kws_times in range(kws_weakup_times):
                if kws_score_np[kws_idx + kws_times][-1] > self.cfg.general.kws_detection_threshold:
                    detected_number += 1

            if detected_number >= kws_weakup_times * self.cfg.general.kws_detection_number_threshold:
                bool_find_kws = True
        
        if bool_find_kws:
            self.params_dict['kws_container_np'] = np.zeros(np.array(kws_score_list).shape)
        else:
            # 存储一定时间的 kws 结果，用于后续滑窗获得结果
            self.params_dict['kws_container_np'] = np.array(kws_score_list)

        return bool_find_kws

    def run_asr(self, contorl_kws_bool=True):
        if not self.bool_do_asr:
            return ''

        asr_string = self.run_asr_normal(contorl_kws_bool)

        if self.cfg.general.asr_second_on:
            if len(asr_string):
                print("Bpe Detect Command: ", asr_string)

                if not contorl_kws_bool:
                    asr_string = self.run_asr_second(contorl_kws_bool)
                    print("Phoneme Detect Command: ", asr_string)

        if len(asr_string):
            asr_string = self.asr_duplicate_check(asr_string)
        return asr_string

    def run_asr_normal(self, contorl_kws_bool=True):
        if not self.bool_do_asr:
            return ''

        # 获取特征
        feature_data_asr = self.params_dict['feature_data_container_np'].astype(np.float32)

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

        # # # debug: cpp & python 一致性测试
        # import cv2
        # # 保存矩阵
        # # cv_file = cv2.FileStorage("/home/huanyuan/share/audio_data/demo_test/test_python.xml", cv2.FILE_STORAGE_WRITE)
        # # cv_file.write('test', net_output)
        # # cv_file.release()

        # # 加载矩阵
        # # cv_file = cv2.FileStorage("/home/huanyuan/share/audio_data/demo_test/test_cpp.xml", cv2.FILE_STORAGE_READ)
        # # net_output = cv_file.getNode("test").mat()
        # # cv_file.release()
        # cv_file = cv2.FileStorage("/home/huanyuan/share/audio_data/demo_test/test_feature_cpp.xml", cv2.FILE_STORAGE_READ)
        # feature_data_fs = cv_file.getNode("feature").mat()
        # cv_file.release()
        # print("feature equal：", (feature_data_fs == feature_data_asr).all())
        
        # decode
        if self.cfg.general.decode_id == 0:
            self.asr_decoder.ctc_decoder(net_output)
        elif self.cfg.general.decode_id == 1:
            self.asr_decoder.beamsearch_decoder(net_output, 5, 0, bswt=1.0, lmwt=0.3)
        else:
            raise Exception("[Unknow:] cfg.general.decode_id = {}".format(self.cfg.general.decode_id))

        if self.cfg.general.language_id == 0:
            self.asr_decoder.show_result_id()
            self.asr_decoder.show_symbol()
            # result_string = self.asr_decoder.output_symbol()

            self.asr_decoder.match_keywords_chinese(self.cfg.general.kws_list, self.cfg.general.kws_dict)
            result_string = self.asr_decoder.output_control_result_string(self.cfg.general.control_kws_list, contorl_kws_bool)
        elif self.cfg.general.language_id == 1:

            self.asr_decoder.show_result_id()
            self.asr_decoder.show_symbol()
            self.asr_decoder.show_symbol_english()

            if self.cfg.general.match_id == 0:
                # bpe
                self.asr_decoder.match_keywords_english_bpe(self.cfg.general.kws_list, 
                                                            self.cfg.general.kws_bpe_dict)
            
            elif self.cfg.general.match_id == 1:
                # phoneme 鲁邦的匹配方式  
                self.asr_decoder.match_keywords_english_phoneme_robust(self.cfg.general.kws_list, 
                                                                        self.cfg.general.kws_phoneme_dict, 
                                                                        self.cfg.general.control_kws_list, 
                                                                        contorl_kws_bool)

            elif self.cfg.general.match_id == 2:
                # phoneme 严格匹配方式
                # self.asr_decoder.match_keywords_english_phoneme_strict(self.cfg.general.kws_list, 
                #                                                         self.cfg.general.kws_phoneme_dict, 
                #                                                         self.cfg.general.control_kws_list, 
                #                                                         contorl_kws_bool, 
                #                                                         self.cfg.general.kws_phoneme_param_dict)
                self.asr_decoder.match_keywords_english_phoneme_strict_simple(self.cfg.general.kws_list, 
                                                                                self.cfg.general.kws_phoneme_dict, 
                                                                                self.cfg.general.control_kws_list, 
                                                                                contorl_kws_bool, 
                                                                                self.cfg.general.kws_phoneme_param_dict)

            elif self.cfg.general.match_id == 3:
                # phoneme 自定义的匹配方式       
                self.asr_decoder.match_keywords_english_phoneme_combine(self.cfg.general.kws_list, 
                                                                        self.cfg.general.kws_phoneme_dict, 
                                                                        self.cfg.general.control_kws_list, 
                                                                        contorl_kws_bool, 
                                                                        self.cfg.general.kws_phoneme_param_dict)

            # result_string = self.asr_decoder.output_result_string()
            result_string = self.asr_decoder.output_control_result_string(self.cfg.general.control_kws_list, 
                                                                            contorl_kws_bool)
        else:
            raise Exception("[Unknow:] cfg.general.language_id = {}".format(self.cfg.general.language_id))

        return result_string

    def run_asr_second(self, contorl_kws_bool=True):
        
        if not self.bool_do_asr:
            return ''

        # 获取特征
        feature_data_asr = self.params_dict['feature_data_container_np'].astype(np.float32)

        # 模型前向传播
        if self.cfg.model.bool_caffe:
            net_output = model.caffe_model_forward(self.asr_net_second, 
                                                    feature_data_asr, 
                                                    self.cfg.model.asr_second_net_input_name, 
                                                    self.cfg.model.asr_second_net_output_name)
            net_output = np.squeeze(net_output)
            net_output = net_output.T
        elif self.cfg.model.bool_pytorch:
            net_output = model.pytorch_model_forward(self.asr_net_second, 
                                                    feature_data_asr, 
                                                    self.bool_gpu)
            net_output = np.squeeze(net_output)
        
        # decode
        if self.cfg.general.second_decode_id == 0:
            self.asr_decoder_second.ctc_decoder(net_output)
        elif self.cfg.general.second_decode_id == 1:
            self.asr_decoder_second.beamsearch_decoder(net_output, 5, 0, bswt=1.0, lmwt=0.3)
        else:
            print("[Unknow:] cfg.general.decode_id. ")

        if self.cfg.general.language_id == 0:
            self.asr_decoder_second.show_result_id()
            self.asr_decoder_second.show_symbol()

            result_string = self.asr_decoder_second.output_symbol()
        elif self.cfg.general.language_id == 1:
            self.asr_decoder_second.show_result_id()
            self.asr_decoder_second.show_symbol()
            self.asr_decoder_second.show_symbol_english()

            if self.cfg.general.second_match_id == 1:
                # phoneme 鲁邦的匹配方式  
                self.asr_decoder_second.match_keywords_english_phoneme_robust(self.cfg.general.kws_list,
                                                                                self.cfg.general.kws_phoneme_dict, 
                                                                                self.cfg.general.control_kws_list, 
                                                                                contorl_kws_bool)

            elif self.cfg.general.second_match_id == 2:
                # phoneme 严格匹配方式
                self.asr_decoder_second.match_keywords_english_phoneme_strict(self.cfg.general.kws_list, 
                                                                                self.cfg.general.kws_phoneme_dict, 
                                                                                self.cfg.general.control_kws_list, 
                                                                                contorl_kws_bool, 
                                                                                self.cfg.general.kws_phoneme_param_dict)

            elif self.cfg.general.second_match_id == 3:
                # phoneme 自定义的匹配方式       
                self.asr_decoder_second.match_keywords_english_phoneme_combine(self.cfg.general.kws_list, 
                                                                                self.cfg.general.kws_phoneme_dict, 
                                                                                self.cfg.general.control_kws_list, 
                                                                                contorl_kws_bool, 
                                                                                self.cfg.general.kws_phoneme_param_dict)

            # result_string = self.asr_decoder_second.output_result_string()
            result_string = self.asr_decoder_second.output_control_result_string(self.cfg.general.control_kws_list, 
                                                                                    contorl_kws_bool)
        else:
            print("[Unknow:] cfg.general.language_id. ")

        return result_string

    def run_sv(self):

        if not self.bool_do_sv:
            return 

        wav = np.array(self.output_dict['output_data_list'])
        wav = wav[ - self.cfg.general.sample_rate * 2:]
        len(wav)
        embedding = sv_model.pytorch_sv_model_forward(self.cfg_sv, self.sv_net, wav, self.bool_gpu)
        self.params_dict['sv_embedding_container'].append(embedding)
        self.params_dict['sv_embedding_container'] = self.params_dict['sv_embedding_container'][-10:]
        if len(self.params_dict['sv_embedding_container']) > 3:
            sv_model.show_embedding(self.params_dict['sv_embedding_container'])
        
        return

    def asr_duplicate_update_counter(self):
        for key in self.params_dict['asr_duplicate_counter']:
            if self.params_dict['asr_duplicate_counter'][key] > 0:
                self.params_dict['asr_duplicate_counter'][key] = self.params_dict['asr_duplicate_counter'][key] - self.cfg.general.window_size_ms
                print(key, self.params_dict['asr_duplicate_counter'][key])
    
    def asr_duplicate_check(self, asr_string):
        res_string = ""
        tmp_string = asr_string.split(' ')
        for idx in range(len(tmp_string)):
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

    def output_wave(self, output_name):
        output_name = '_'.join(str(output_name).strip().split(' '))
        # save audio
        if self.cfg.general.bool_output_wave:
            date_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            output_path = os.path.join(self.cfg.test.output_folder, '{}_{}_{}.wav'.format(output_name, date_time, self.output_dict['output_kws_id']))
            # wave_loader = WaveLoader_Python.WaveLoader_Librosa(self.cfg.general.sample_rate)
            wave_loader = WaveLoader_Python.WaveLoader_Soundfile(self.cfg.general.sample_rate)
            self.output_dict['output_kws_id'] += 1
            wave_loader.save_data(np.array(self.output_dict['output_data_list']).astype(np.int16), output_path)