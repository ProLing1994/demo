import copy
import heapq
import numpy as np
import kenlm


class Ken_LM:
    
    def __init__(self, lmPATH):
        self.__model = kenlm.Model(lmPATH)
        self.senlen = 3

    def compute(self, state, word):
        assert isinstance(state, list)
        state.append(word)
        sentence = " ".join(state)
        # 输入需要显式指定 <s> 起始符，不会默认添加，然后忽略 eos 终止符，不会为输入的 sentence 添加默认终止符 </s>
        # list(self.__model.full_scores(sentence, bos=False, eos=False))[-1][0] 获取最后一个时刻的得分（而不是将每个时刻的得分累加）
        # prob = self.__model.score(sentence, bos=True, eos=True)
        # prob = self.__model.score(sentence, bos=False, eos=False)
        prob = list(self.__model.full_scores(sentence, bos=False, eos=False))[-1][0]

        # print(self.__model.score(sentence, bos=True, eos=True))
        # print(self.__model.score(sentence, bos=False, eos=False))
        # print(list(self.__model.full_scores(sentence, bos=True, eos=True)))
        # print(list(self.__model.full_scores(sentence, bos=True, eos=False)))
        # print(list(self.__model.full_scores(sentence, bos=False, eos=True)))
        # print(list(self.__model.full_scores(sentence, bos=False, eos=False)))
        # print(list(self.__model.full_scores(sentence, bos=False, eos=False))[-1][0])
        if len(state) < self.__model.order:
            return state, prob
        else:
            return state[1:], prob


def get_edit_distance(sentence1, sentence2):
    '''
    :param sentence1: sentence1 list
    :param sentence2: sentence2 list
    :return: distence between sentence1 and sentence2
    '''
    len1 = len(sentence1)
    len2 = len(sentence2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if sentence1[i-1] == sentence2[j-1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta,
                           min(dp[i-1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


def remove_symbol_number(word):
    res_word = ''
    for idx in range(len(word)):
        if word[idx] >= '0' and word[idx] <= '9':
            continue
        else:
            res_word += word[idx]
    return res_word


def match_symbol_verb(key_symbol, input_symbol):
    key_word_without_number = remove_symbol_number(key_symbol)
    symbol_word_without_number = remove_symbol_number(input_symbol)

    if len(symbol_word_without_number) < len(key_word_without_number):
        return False
    
    dist = get_edit_distance(key_word_without_number, symbol_word_without_number[:len(key_word_without_number)]) 

    if dist == 0:
        return True
    else:
        return False

def match_symbol_noum(key_symbol, input_symbol):
    key_word_without_number = remove_symbol_number(key_symbol)
    symbol_word_without_number = remove_symbol_number(input_symbol)

    dist = get_edit_distance(key_word_without_number, symbol_word_without_number)
    if dist == 0 or (dist < 2 and len(key_word_without_number) > 4) or (dist < 3 and len(key_word_without_number) > 6):
        return True
    else:
        return match_symbol_verb(key_symbol, input_symbol)

def match_phoneme_verb(key_phonme, input_phonme):
    assert len(key_phonme) >= len(input_phonme)

    dist = get_edit_distance(key_phonme, input_phonme) 

    if dist == 0:
        return True
    else:
        return False

def match_phoneme_noum(key_phonme, input_phonme):
    assert len(key_phonme) >= len(input_phonme)

    dist = get_edit_distance(key_phonme, input_phonme) 

    # 匹配策略：名词编辑距离小于阈值
    # if dist <= 1:
    if dist <= 1 or (dist < 3 and len(key_phonme) > 6):
        return True
    else:
        return False

class Decode(object):
    """ decode python wrapper """

    def __init__(self):
        self.id2word = []
        self.result_id = []
        self.result_string = []

        # lm
        self.lm = None
        self.word_bs = []
        self.verb_socres_threshold = -0.2

    def init_symbol_list(self, asr_dict_path):
        with open(asr_dict_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.id2word.append(line.strip())

    def init_lm_model(self, asr_lm_path):
        self.lm = Ken_LM(asr_lm_path)

    def show_result_id(self):
        print(self.result_id)

    def show_symbol(self):
        output_symbol = ""
        for idx in range(len(self.result_id)):
            symbol = self.id2word[self.result_id[idx]]
            output_symbol += symbol + " "
        print(output_symbol)

    def show_symbol_english(self):
        output_symbol = ""
        for idx in range(len(self.result_id)):
            symbol = self.id2word[self.result_id[idx]]

            if symbol[0] == '_':
                if idx != 0:
                    output_symbol += " "
                output_symbol += symbol[1:]
            else:
                output_symbol += symbol
        print(output_symbol)

    def output_symbol(self):
        output_symbol = ""
        for idx in range(len(self.result_id)):
            symbol = self.id2word[self.result_id[idx]]

            output_symbol += symbol + " "
        return output_symbol

    def output_symbol_english(self):
        output_symbol = ""
        for idx in range(len(self.result_id)):
            symbol = self.id2word[self.result_id[idx]]

            if symbol[0] == '_':
                if idx != 0:
                    output_symbol += " "
                output_symbol += symbol[1:]
            else:
                output_symbol += symbol
        if len(self.result_id):
            output_symbol += " "
        return output_symbol
        
    def output_result_string(self):
        result_string = ''
        for idx in range(len(self.result_string)):
            result_string += self.result_string[idx]
            result_string += ' '
        return result_string.strip()

    def output_control_result_string(self, control_kws_list, contorl_kws_bool=True):
        result_string = ''

        for idx in range(len(self.result_string)):
            find_control_kws = self.result_string[idx] in control_kws_list

            if contorl_kws_bool and find_control_kws:
                result_string += "control_" + self.result_string[idx]
                result_string += ' '
            elif contorl_kws_bool and not find_control_kws:
                result_string += self.result_string[idx]
                result_string += ' '
            elif not contorl_kws_bool and not find_control_kws:
                result_string += self.result_string[idx]
                result_string += ' '
        return result_string.strip()

    def phoneme2symbol(self, phoneme_list):
        output_symbol = ""
        for idx in range(len(phoneme_list)):
            symbol = phoneme_list[idx]
            
            if symbol == "":
                continue

            if symbol[0] == '_':
                if idx != 0:
                    output_symbol += " "
                output_symbol += symbol[1:]
            else:
                output_symbol += symbol
        return output_symbol

    def phonemegroup2phonemelist(self, phoneme_group, phoneme_list, temp_list=[], id = 0):
        if id == len(phoneme_group):
            phoneme_list.append(temp_list.copy())
            return

        for idx in range(len(phoneme_group[id])):
            temp_list.append(phoneme_group[id][idx])
            self.phonemegroup2phonemelist(phoneme_group, phoneme_list, temp_list, id + 1)
            temp_list.pop()
        return 

    def match_keywords_english_bpe(self, kws_list, kws_dict):
        # init
        self.result_string = []
        matching_lable_list = []            # 容器，记录匹配的 label

        # init matching_state_list，匹配状态容器
        matching_state_list = []            # {'words':[], 'lable':[], 'length':0, 'state_id':-1}
        for idx in range(len(kws_list)):
            kws_idx = kws_list[idx]
            for idy in range(len(kws_dict[kws_idx])):
                matching_state_dict = {}
                matching_state_dict['words'] = kws_dict[kws_idx][idy].strip().split(" ")
                matching_state_dict['lable'] = kws_idx
                matching_state_dict['length'] = len(matching_state_dict['words'])
                matching_state_dict['state_id'] = -1
                matching_state_list.append(matching_state_dict)

        # init english_symbol_list
        english_symbol_list = self.output_symbol_english().split(' ')

        # 遍历 english_symbol_list
        for idx in range(len(english_symbol_list)):
            # 遍历 matching_state_list
            for idy in range(len(matching_state_list)):
                # init
                match_bool = False
                words = matching_state_list[idy]['words']
                lable = matching_state_list[idy]['lable']
                length = matching_state_list[idy]['length']
                state_id = matching_state_list[idy]['state_id']

                # 当前策略：
                # 动词：任意匹配 [ing, ed, s]；
                if state_id + 1 == 0:
                    match_bool = match_symbol_verb(words[state_id + 1], english_symbol_list[idx])
                # 名词：编辑距离小于阈值或者任意匹配 [ing, ed, s]
                elif state_id + 1 < length:
                    match_bool = match_symbol_noum(words[state_id + 1], english_symbol_list[idx])
                else:
                    continue

                if match_bool:
                    # 更新 state_id
                    state_id = state_id + 1
                    matching_state_list[idy]['state_id'] = state_id

                    # # 查询匹配成功的字符
                    # print("匹配成功字符：", english_symbol_list[idx], ": ", words[state_id]);
                    # print("匹配成功长度：", state_id + 1, "/", length);

                    if state_id + 1 == length:
                        find_matching_lable_bool = True if lable in matching_lable_list else False

                        if not find_matching_lable_bool:
                            matching_lable_list.append(lable)
                            self.result_string.append(lable)
        return 

    def match_keywords_english_phoneme_strict(self, kws_list, kws_phoneme_dict, control_kws_list, contorl_kws_bool, kws_phoneme_param_dict={}):
        # init
        self.result_string = []
        matching_lable_list = []            # 容器，记录匹配的 label
        matched_verb_id = []                # 容器，记录动词的 id

        # init matching_state_list，匹配状态容器
        matching_state_list = []            # {'phoneme_groups':[], 'lable':[], 'length':0, 'state_id':-1, 'matched_id':-1, 'matched_interval':0, 'verb_socres_threshold':-0.2}
        for idx in range(len(kws_list)):
            kws_idx = kws_list[idx]
            matching_state_dict = {}
            matching_state_dict['phoneme_groups'] = kws_phoneme_dict[kws_idx]
            matching_state_dict['lable'] = kws_idx
            matching_state_dict['length'] = len(matching_state_dict['phoneme_groups'])
            matching_state_dict['state_id'] = -1
            matching_state_dict['matched_id'] = -1
            matching_state_dict['matched_interval'] = 0
            matching_state_dict['verb_socres_threshold'] = kws_phoneme_param_dict[kws_idx]["verb_socres_threshold"] if kws_idx in kws_phoneme_param_dict else self.verb_socres_threshold 
            matching_state_list.append(matching_state_dict)

        english_phoneme_list = [self.id2word[idx] for idx in self.result_id]

        # 遍历 english_phoneme_list
        for idx in range(len(english_phoneme_list)):
            # 遍历 matching_state_list
            for idy in range(len(matching_state_list)):
                # init
                match_bool = False
                phoneme_groups = matching_state_list[idy]['phoneme_groups']
                lable = matching_state_list[idy]['lable']
                length = matching_state_list[idy]['length']
                state_id = matching_state_list[idy]['state_id']
                matched_id = matching_state_list[idy]['matched_id']
                matched_interval = matching_state_list[idy]['matched_interval']
                verb_socres_threshold = matching_state_list[idy]['verb_socres_threshold']

                # check 
                if idx < matched_id:
                    continue

                find_control_kws = lable in control_kws_list
                if contorl_kws_bool and not find_control_kws:
                    continue
                if not contorl_kws_bool and find_control_kws:
                    continue

                # 当前策略：
                # 动词：任意匹配 [ing, ed, s]；
                if state_id + 1 == 0:

                    # 匹配规则：
                    # 1、匹配动词音素：必须以 "_" 开头，动词任意匹配 [ing, ed, s]，动词音素的编辑距离等于 0 
                    if '_' not in english_phoneme_list[idx]:
                        continue

                    # 生成 phoneme_list 
                    phoneme_list = []
                    self.phonemegroup2phonemelist(phoneme_groups[state_id + 1], phoneme_list)

                    for idz in range(len(phoneme_list)):
                        key_phoneme = [i for i in phoneme_list[idz] if i != ""]
                        key_phoneme = ' '.join(key_phoneme).strip().split(" ")
                        key_symbol = self.phoneme2symbol(key_phoneme)
                        input_symbol = self.phoneme2symbol(english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))])

                        # 动词匹配
                        match_bool = match_phoneme_verb(key_phoneme, english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))])

                        # 动词匹配成功
                        if (match_bool):
                            # 查询匹配的字符
                            print("动词匹配字符：", english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))], ": ", key_phoneme)
                            print("动词匹配字符：", input_symbol, ": ", key_symbol)

                            # 计算动词平均得分（模型得分），若均值大于阈值（-0.2），则认为匹配成功
                            # 记录匹配成功的动词的起止位置
                            matched_verb_id = range(idx, min(len(english_phoneme_list), idx + len(key_phoneme)))

                            if len(self.word_bs):
                                verb_socres = 0.0
                                for verb_id in range(len(matched_verb_id)):
                                    print("动词字符：", self.word_bs[matched_verb_id[verb_id]][0], "动词得分: ", self.word_bs[matched_verb_id[verb_id]][1])
                                    verb_socres += self.word_bs[matched_verb_id[verb_id]][1]
                                verb_socres /= len(matched_verb_id)
                                print("动词平均得分（模型得分）：", verb_socres)

                                if verb_socres < verb_socres_threshold:
                                    print("动词匹配失败 ...")
                                    match_bool = False
                                else:
                                    print("动词匹配成功 !!!")
                            
                            if (match_bool):
                                matched_id = idx + len(key_phoneme)
                                matching_state_list[idy]['matched_id'] = matched_id
                                break

                # 名词：编辑距离小于 1 或者任意匹配 [ing, ed, s]
                elif state_id + 1 < length:

                    # 匹配规则
                    # 2、匹配名词音素：名词不必紧跟动词（允许有相应的间隔），名词音素的编辑距离小于阈值

                    # 生成 phoneme_list
                    phoneme_list = []
                    self.phonemegroup2phonemelist(phoneme_groups[state_id + 1], phoneme_list)

                    for idz in range(len(phoneme_list)):
                        key_phoneme = [i for i in phoneme_list[idz] if i != ""]
                        key_phoneme = ' '.join(key_phoneme).strip().split(" ")
                        key_symbol = self.phoneme2symbol(key_phoneme)
                        input_symbol = self.phoneme2symbol(english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))])

                        # 名词匹配 
                        match_bool = False
                        # 必须以 "_" 开头 
                        if '_' in english_phoneme_list[idx] and matched_interval < 1:
                            match_bool = match_phoneme_noum(key_phoneme, english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))])
                        
                        if (match_bool):
                            # 查询匹配的字符
                            print("名词匹配字符：", english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))], ": ", key_phoneme)
                            print("名词匹配字符：", input_symbol, ": ", key_symbol)

                            matched_id = idx + len(key_phoneme)
                            matching_state_list[idy]['matched_id'] = matched_id
                            break
                    
                    if '_' in english_phoneme_list[idx]:
                        matched_interval = matched_interval + 1
                        matching_state_list[idy]['matched_interval'] = matched_interval

                else:
                    continue

                if match_bool:
                    # 更新 state_id
                    state_id = state_id + 1
                    matching_state_list[idy]['state_id'] = state_id

                    if state_id + 1 == length:
                        find_matching_lable_bool = True if lable in matching_lable_list else False

                        if not find_matching_lable_bool:
                            matching_lable_list.append(lable)
                            self.result_string.append(lable)

                        if 'mute_audio' in matching_lable_list and 'unmute_audio' in matching_lable_list:
                            self.result_string = []
                            matching_lable_list = []            # 容器，记录匹配的 label
                            self.result_string.append('unmute_audio')    
                            matching_lable_list.append('unmute_audio')
        return

    def match_keywords_english_phoneme_strict_simple(self, kws_list, kws_phoneme_dict, control_kws_list, contorl_kws_bool, kws_phoneme_param_dict={}):
        # init
        self.result_string = []
        matching_lable_list = []            # 容器，记录匹配的 label
        matched_verb_id = []                # 容器，记录动词的 id

        # init matching_state_list，匹配状态容器
        matching_state_list = []            # {'phoneme_groups':[], 'lable':[], 'length':0, 'state_id':-1, 'matched_id':-1, 'matched_interval':0, 'verb_socres_threshold':-0.2}
        for idx in range(len(kws_list)):
            kws_idx = kws_list[idx]
            matching_state_dict = {}
            matching_state_dict['phoneme_groups'] = kws_phoneme_dict[kws_idx]
            matching_state_dict['lable'] = kws_idx
            matching_state_dict['length'] = len(matching_state_dict['phoneme_groups'])
            matching_state_dict['state_id'] = -1
            matching_state_dict['matched_id'] = -1
            matching_state_dict['matched_interval'] = 0
            matching_state_dict['verb_socres_threshold'] = kws_phoneme_param_dict[kws_idx]["verb_socres_threshold"] if kws_idx in kws_phoneme_param_dict else self.verb_socres_threshold 
            matching_state_list.append(matching_state_dict)

        english_phoneme_list = [self.id2word[idx] for idx in self.result_id]

        # 遍历 english_phoneme_list
        for idx in range(len(english_phoneme_list)):
            # 遍历 matching_state_list
            for idy in range(len(matching_state_list)):
                # init
                match_bool = False
                phoneme_groups = matching_state_list[idy]['phoneme_groups']
                lable = matching_state_list[idy]['lable']
                length = matching_state_list[idy]['length']
                state_id = matching_state_list[idy]['state_id']
                matched_id = matching_state_list[idy]['matched_id']
                matched_interval = matching_state_list[idy]['matched_interval']
                verb_socres_threshold = matching_state_list[idy]['verb_socres_threshold']

                # check 
                if idx < matched_id:
                    continue

                find_control_kws = lable in control_kws_list
                if contorl_kws_bool and not find_control_kws:
                    continue
                if not contorl_kws_bool and find_control_kws:
                    continue

                # 当前策略：
                # 动词：任意匹配 [ing, ed, s]；
                if state_id + 1 == 0:

                    # 匹配规则：
                    # 1、匹配动词音素：必须以 "_" 开头，动词任意匹配 [ing, ed, s]，动词音素的编辑距离等于 0 
                    if '_' not in english_phoneme_list[idx]:
                        continue

                    # 生成 phoneme_list 
                    phoneme_list = [phoneme_groups[state_id + 1]]

                    for idz in range(len(phoneme_list)):
                        key_phoneme = [i for i in phoneme_list[idz] if i != ""]
                        key_phoneme = ' '.join(key_phoneme).strip().split(" ")
                        key_symbol = self.phoneme2symbol(key_phoneme)
                        input_symbol = self.phoneme2symbol(english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))])

                        # 动词匹配
                        match_bool = match_phoneme_verb(key_phoneme, english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))])

                        # 动词匹配成功
                        if (match_bool):
                            # 查询匹配的字符
                            print("动词匹配字符：", english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))], ": ", key_phoneme)
                            print("动词匹配字符：", input_symbol, ": ", key_symbol)

                            # 计算动词平均得分（模型得分），若均值大于阈值（-0.2），则认为匹配成功
                            # 记录匹配成功的动词的起止位置
                            matched_verb_id = range(idx, min(len(english_phoneme_list), idx + len(key_phoneme)))

                            if len(self.word_bs):
                                verb_socres = 0.0
                                for verb_id in range(len(matched_verb_id)):
                                    print("动词字符：", self.word_bs[matched_verb_id[verb_id]][0], "动词得分: ", self.word_bs[matched_verb_id[verb_id]][1])
                                    verb_socres += self.word_bs[matched_verb_id[verb_id]][1]
                                verb_socres /= len(matched_verb_id)
                                print("动词平均得分（模型得分）：", verb_socres)

                                if verb_socres < verb_socres_threshold:
                                    print("动词匹配失败 ...")
                                    match_bool = False
                                else:
                                    print("动词匹配成功 !!!")
                            
                            if (match_bool):
                                matched_id = idx + len(key_phoneme)
                                matching_state_list[idy]['matched_id'] = matched_id
                                break

                # 名词：编辑距离小于 1 或者任意匹配 [ing, ed, s]
                elif state_id + 1 < length:

                    # 匹配规则
                    # 2、匹配名词音素：名词不必紧跟动词（允许有相应的间隔），名词音素的编辑距离小于阈值

                    # 生成 phoneme_list
                    phoneme_list = [phoneme_groups[state_id + 1]]

                    for idz in range(len(phoneme_list)):
                        key_phoneme = [i for i in phoneme_list[idz] if i != ""]
                        key_phoneme = ' '.join(key_phoneme).strip().split(" ")
                        key_symbol = self.phoneme2symbol(key_phoneme)
                        input_symbol = self.phoneme2symbol(english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))])

                        # 名词匹配 
                        match_bool = False
                        # 必须以 "_" 开头 
                        if '_' in english_phoneme_list[idx] and matched_interval < 1:
                            match_bool = match_phoneme_noum(key_phoneme, english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))])
                        
                        if (match_bool):
                            # 查询匹配的字符
                            print("名词匹配字符：", english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))], ": ", key_phoneme)
                            print("名词匹配字符：", input_symbol, ": ", key_symbol)

                            matched_id = idx + len(key_phoneme)
                            matching_state_list[idy]['matched_id'] = matched_id
                            break
                    
                    if '_' in english_phoneme_list[idx]:
                        matched_interval = matched_interval + 1
                        matching_state_list[idy]['matched_interval'] = matched_interval

                else:
                    continue

                if match_bool:
                    # 更新 state_id
                    state_id = state_id + 1
                    matching_state_list[idy]['state_id'] = state_id

                    if state_id + 1 == length:
                        find_matching_lable_bool = True if lable in matching_lable_list else False

                        if not find_matching_lable_bool:
                            matching_lable_list.append(lable)
                            self.result_string.append(lable)

                        if 'mute_audio' in matching_lable_list and 'unmute_audio' in matching_lable_list:
                            self.result_string = []
                            matching_lable_list = []            # 容器，记录匹配的 label
                            self.result_string.append('unmute_audio')    
                            matching_lable_list.append('unmute_audio')
        return

    def match_keywords_english_phoneme_robust(self, kws_list, kws_phoneme_dict, control_kws_list, contorl_kws_bool):
        # init
        self.result_string = []
        matching_lable_list = []            # 容器，记录匹配的 label

        # init matching_state_list，匹配状态容器
        matching_state_list = []            # {'phoneme_groups':[], 'lable':[], 'length':0, 'state_id':-1, 'matched_id':-1}
        for idx in range(len(kws_list)):
            kws_idx = kws_list[idx]
            matching_state_dict = {}
            matching_state_dict['phoneme_groups'] = kws_phoneme_dict[kws_idx]
            matching_state_dict['lable'] = kws_idx
            matching_state_dict['length'] = len(matching_state_dict['phoneme_groups'])
            matching_state_dict['state_id'] = -1
            matching_state_dict['matched_id'] = -1
            matching_state_list.append(matching_state_dict)

        english_phoneme_list = [self.id2word[idx] for idx in self.result_id]

        # 遍历 english_phoneme_list
        for idx in range(len(english_phoneme_list)):
            # 遍历 matching_state_list
            for idy in range(len(matching_state_list)):
                # init
                match_bool = False
                phoneme_groups = matching_state_list[idy]['phoneme_groups']
                lable = matching_state_list[idy]['lable']
                length = matching_state_list[idy]['length']
                state_id = matching_state_list[idy]['state_id']
                matched_id = matching_state_list[idy]['matched_id']

                # check 
                if idx < matched_id:
                    continue

                find_control_kws = lable in control_kws_list
                if contorl_kws_bool and not find_control_kws:
                    continue
                if not contorl_kws_bool and find_control_kws:
                    continue

                # 当前策略：
                # 动词：任意匹配 [ing, ed, s]；
                if state_id + 1 == 0:

                    # 匹配规则：
                    # 1、匹配动词音素：动词任意匹配 [ing, ed, s]，动词音素的编辑距离等于 0 
                    if '_' not in english_phoneme_list[idx]:
                        continue

                    # 生成 phoneme_list 
                    phoneme_list = []
                    self.phonemegroup2phonemelist(phoneme_groups[state_id + 1], phoneme_list)

                    for idz in range(len(phoneme_list)):
                        key_phoneme = [i for i in phoneme_list[idz] if i != ""]
                        key_phoneme = ' '.join(key_phoneme).strip().split(" ")
                        key_symbol = self.phoneme2symbol(key_phoneme)
                        input_symbol = self.phoneme2symbol(english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))])

                        # 动词匹配
                        match_bool = match_symbol_verb(key_symbol, input_symbol)

                        # 动词匹配成功
                        if (match_bool):
                            # 查询匹配的字符
                            print("动词匹配字符：", english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))], ": ", key_phoneme)
                            print("动词匹配字符：", input_symbol, ": ", key_symbol)

                            matched_id = idx + len(key_phoneme)
                            matching_state_list[idy]['matched_id'] = matched_id
                            break

                # 名词：编辑距离小于 1 或者任意匹配 [ing, ed, s]
                elif state_id + 1 < length:

                    # 匹配规则
                    # 2、匹配名词音素：名词不必紧跟动词（允许有相应的间隔），名词音素的编辑距离小于 1

                    # 生成 phoneme_list
                    phoneme_list = []
                    self.phonemegroup2phonemelist(phoneme_groups[state_id + 1], phoneme_list)

                    for idz in range(len(phoneme_list)):
                        key_phoneme = [i for i in phoneme_list[idz] if i != ""]
                        key_phoneme = ' '.join(key_phoneme).strip().split(" ")
                        key_symbol = self.phoneme2symbol(key_phoneme)
                        input_symbol = self.phoneme2symbol(english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))])
                        
                        # 名词匹配 
                        match_bool = match_symbol_noum(key_symbol, input_symbol)
                        
                        if (match_bool):
                            # 查询匹配的字符
                            print("名词匹配字符：", english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))], ": ", key_phoneme)
                            print("名词匹配字符：", input_symbol, ": ", key_symbol)

                            matched_id = idx + len(key_phoneme)
                            matching_state_list[idy]['matched_id'] = matched_id
                            break

                if match_bool:
                    # 更新 state_id
                    state_id = state_id + 1
                    matching_state_list[idy]['state_id'] = state_id

                    if state_id + 1 == length:
                        find_matching_lable_bool = True if lable in matching_lable_list else False

                        if not find_matching_lable_bool:
                            matching_lable_list.append(lable)
                            self.result_string.append(lable)

                        if 'mute_audio' in matching_lable_list and 'unmute_audio' in matching_lable_list:
                            self.result_string = []
                            matching_lable_list = []            # 容器，记录匹配的 label
                            self.result_string.append('unmute_audio')    
                            matching_lable_list.append('unmute_audio')
        return

    def match_keywords_english_phoneme_combine(self, kws_list, kws_phoneme_dict, control_kws_list, contorl_kws_bool, kws_phoneme_param_dict={}):
        # init
        self.result_string = []
        matching_lable_list = []            # 容器，记录匹配的 label
        matched_verb_id = []                # 容器，记录动词的 id

        # init matching_state_list，匹配状态容器
        matching_state_list = []            # {'phoneme_groups':[], 'lable':[], 'length':0, 'state_id':-1, 'matched_id':-1, 'matched_interval':0, 'verb_socres_threshold':-0.2, 'strategy': strict}
        for idx in range(len(kws_list)):
            kws_idx = kws_list[idx]
            matching_state_dict = {}
            matching_state_dict['phoneme_groups'] = kws_phoneme_dict[kws_idx]
            matching_state_dict['lable'] = kws_idx
            matching_state_dict['length'] = len(matching_state_dict['phoneme_groups'])
            matching_state_dict['state_id'] = -1
            matching_state_dict['matched_id'] = -1
            matching_state_dict['matched_interval'] = 0
            matching_state_dict['verb_socres_threshold'] = kws_phoneme_param_dict[kws_idx]["verb_socres_threshold"] if kws_idx in kws_phoneme_param_dict else self.verb_socres_threshold 
            matching_state_dict['strategy'] = kws_phoneme_param_dict[kws_idx]["strategy"] if kws_idx in kws_phoneme_param_dict else 'strict'
            matching_state_list.append(matching_state_dict)

        english_phoneme_list = [self.id2word[idx] for idx in self.result_id]
        # english_phoneme_list = ['_K', 'IY1', 'P', '_L', 'u', '_Y', 'AO1', 'R', '_HH', 'AE1', 'N', 'D']
        
        # 遍历 english_phoneme_list
        for idx in range(len(english_phoneme_list)):
            # 遍历 matching_state_list
            for idy in range(len(matching_state_list)):
                # init
                match_bool = False
                phoneme_groups = matching_state_list[idy]['phoneme_groups']
                lable = matching_state_list[idy]['lable']
                length = matching_state_list[idy]['length']
                state_id = matching_state_list[idy]['state_id']
                matched_id = matching_state_list[idy]['matched_id']
                matched_interval = matching_state_list[idy]['matched_interval']
                verb_socres_threshold = matching_state_list[idy]['verb_socres_threshold']
                strategy = matching_state_list[idy]['strategy']

                # check 
                if idx < matched_id:
                    continue
                
                # 判断是否为控制词，减少不需要的匹配
                find_control_kws = lable in control_kws_list
                if contorl_kws_bool and not find_control_kws:
                    continue
                if not contorl_kws_bool and find_control_kws:
                    continue

                # 当前策略：
                # 动词：任意匹配 [ing, ed, s]；
                if state_id + 1 == 0:

                    # 匹配规则：
                    # 1、匹配动词音素：必须以 "_" 开头，动词任意匹配 [ing, ed, s]，动词音素的编辑距离等于 0 
                    if '_' not in english_phoneme_list[idx]:
                        continue

                    # 生成 phoneme_list 
                    phoneme_list = []
                    self.phonemegroup2phonemelist(phoneme_groups[state_id + 1], phoneme_list)

                    for idz in range(len(phoneme_list)):
                        key_phoneme = [i for i in phoneme_list[idz] if i != ""]
                        key_phoneme = ' '.join(key_phoneme).strip().split(" ")
                        key_symbol = self.phoneme2symbol(key_phoneme)
                        input_symbol = self.phoneme2symbol(english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))])

                        # print(key_phoneme, key_symbol)
                        # 动词匹配
                        if strategy == "robust":
                            match_bool = match_symbol_verb(key_symbol, input_symbol)
                        elif strategy == "strict":
                            match_bool = match_phoneme_verb(key_phoneme, english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))])

                        # 动词匹配成功
                        if (match_bool):
                            # 查询匹配的字符
                            print("动词匹配字符：", english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))], ": ", key_phoneme)
                            print("动词匹配字符：", input_symbol, ": ", key_symbol)

                            if strategy == "strict":

                                # 计算动词平均得分（模型得分），若均值大于阈值（-0.2），则认为匹配成功
                                # 记录匹配成功的动词的起止位置
                                matched_verb_id = range(idx, min(len(english_phoneme_list), idx + len(key_phoneme)))

                                if len(self.word_bs):
                                    verb_socres = 0.0
                                    for verb_id in range(len(matched_verb_id)):
                                        print("动词字符：", self.word_bs[matched_verb_id[verb_id]][0], "动词得分: ", self.word_bs[matched_verb_id[verb_id]][1])
                                        verb_socres += self.word_bs[matched_verb_id[verb_id]][1]
                                    verb_socres /= len(matched_verb_id)
                                    print("动词平均得分（模型得分）：", verb_socres)

                                    if verb_socres < verb_socres_threshold:
                                        print("动词匹配失败 ...")
                                        match_bool = False
                                    else:
                                        print("动词匹配成功 !!!")
                            
                            if (match_bool):
                                matched_id = idx + len(key_phoneme)
                                matching_state_list[idy]['matched_id'] = matched_id
                                break

                # 名词：编辑距离小于 1 或者任意匹配 [ing, ed, s]
                elif state_id + 1 < length:

                    # 匹配规则
                    # 2、匹配名词音素：名词不必紧跟动词（允许有相应的间隔），名词音素的编辑距离小于 1

                    # 生成 phoneme_list
                    phoneme_list = []
                    self.phonemegroup2phonemelist(phoneme_groups[state_id + 1], phoneme_list)

                    for idz in range(len(phoneme_list)):
                        key_phoneme = [i for i in phoneme_list[idz] if i != ""]
                        key_phoneme = ' '.join(key_phoneme).strip().split(" ")
                        key_symbol = self.phoneme2symbol(key_phoneme)
                        input_symbol = self.phoneme2symbol(english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))])

                        # 名词匹配 
                        if strategy == "robust":
                            match_bool = match_symbol_noum(key_symbol, input_symbol)
                        elif strategy == "strict":
                            match_bool = False
                            if '_' in english_phoneme_list[idx] and matched_interval < 1:
                                match_bool = match_phoneme_noum(key_phoneme, english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))])
                        
                        if (match_bool):
                            # 查询匹配的字符
                            print("名词匹配字符：", english_phoneme_list[idx: min(len(english_phoneme_list), idx + len(key_phoneme))], ": ", key_phoneme)
                            print("名词匹配字符：", input_symbol, ": ", key_symbol)

                            matched_id = idx + len(key_phoneme)
                            matching_state_list[idy]['matched_id'] = matched_id
                            break
                    
                    if '_' in english_phoneme_list[idx]:
                        matched_interval = matched_interval + 1
                        matching_state_list[idy]['matched_interval'] = matched_interval

                else:
                    continue

                if match_bool:
                    # 更新 state_id
                    state_id = state_id + 1
                    matching_state_list[idy]['state_id'] = state_id

                    if state_id + 1 == length:
                        # 更新输出，保证每类关键词只输出一次
                        find_matching_lable_bool = True if lable in matching_lable_list else False

                        if not find_matching_lable_bool:
                            matching_lable_list.append(lable)
                            self.result_string.append(lable)

                        if 'mute_audio' in matching_lable_list and 'unmute_audio' in matching_lable_list:
                            self.result_string = []
                            matching_lable_list = []            # 容器，记录匹配的 label
                            self.result_string.append('unmute_audio')    
                            matching_lable_list.append('unmute_audio')
        return

    def match_keywords_chinese(self, kws_list, kws_dict):
        # init 
        index = 0
        match_id_list = []
        self.result_string = []
        
        symbol_list = [self.id2word[self.result_id[idx]] for idx in range(len(self.result_id))]
        while index < len(symbol_list):
            # init 
            match_flag = False

            for kws_idx in range(len(kws_list)):
                keywords_list = kws_dict[kws_list[kws_idx]].split(' ')
                if (symbol_list[index] == keywords_list[0] and (index + len(keywords_list)) <= len(symbol_list)):
                    tmp_symbol = [symbol_list[index + idx] for idx in range(len(keywords_list))]
                    dist = get_edit_distance(tmp_symbol, keywords_list)

                    if (dist == 0 or (dist < 2 and len(keywords_list) > 4) or (dist < 3 and len(keywords_list) > 6)):
                        match_flag = True
                        match_id_list.append(kws_idx)
                        break

            if (match_flag):
                index = index + len(kws_dict[kws_list[match_id_list[-1]]].split(' ')) - 1
            else:
                index += 1

        if len(match_id_list):
            for idx in range(len(match_id_list)):
                self.result_string.append(kws_list[match_id_list[idx]])

    def ctc_decoder(self, input_data):
        # init
        self.result_id = []
        blank_id = 0
        last_max_id = 0
        frame_num = input_data.shape[0]
        feature_num = input_data.shape[1]

        for idx in range(frame_num):
            max_value = input_data[idx][0]
            max_id = 0

            for idy in range(feature_num):
                if input_data[idx][idy] > max_value:
                    max_value = input_data[idx][idy]
                    max_id = idy
            
            if max_id != blank_id and last_max_id != max_id:
                self.result_id.append(max_id)
                last_max_id = max_id
                # print("id: ", max_id, ", value: ", max_value)
        return 

    def beamsearch_decoder(self, prob, beamSize, blankID, bswt=1.0, lmwt=0.3, beams_topk=[]):
        '''
        # post-processing ( append </s> symbol and update LM score )
        for one in results:
            newState, lmScore = lm.compute( state=one["lmState"], word="</s>" )
            one["lm"] += lmwt * lmScore
            numFrames = len(one["ali"])
            numWords = len(one["words"]) + 1
            one["total"] = one["bs"] / numFrames + one["lm"] / numWords (对数相加)
            # discard <s>
            one["words"] = " ".join( one["words"] )
            one["lmState"] = newState
        '''
        # 取对数，np.log，用于和语言模型得分相加
        prob = np.log(prob)

        # init
        frame_num = prob.shape[0]

        # initilize LM score (一定要用一个非0的值来初始化，我根据经验，使用未知词的概率来初始化。)
        _, init_lm_score = self.lm.compute(state=[], word="UNK")
        init_lm_score *= lmwt
        if(len(beams_topk) == 0):
            beams_topk = [{"words": [], "ali":[], "result_id":[], "word_bs":[], "lmState":["<s>"], "bs":0, "lm":init_lm_score, "total":0}]
            # print("init_lm_score: ", init_lm_score)

        for i in range(frame_num):
            tmp_beams = []

            # get topk
            score_list = prob[i]
            id_socre = zip(range(len(score_list)), score_list)
            frame_topk = heapq.nlargest(5, id_socre, key=lambda x: x[1])

            for preResult in beams_topk:

                for (id, bs_score) in frame_topk:

                    if(bs_score < -1.6):
                        continue
                    
                    # print("id: ", id, ", bs_score: ", bs_score)
                    tmp_beam = copy.deepcopy(preResult)

                    # 1. compute LM score
                    # ignore blank
                    if id != blankID:  
                        # ignore continuous repeated symbols
                        if len(tmp_beam["ali"]) == 0 or id != tmp_beam["ali"][-1]:
                            symbol = self.id2word[id]
                            newState, lmScore = self.lm.compute(state=tmp_beam["lmState"], word=symbol)
                            tmp_beam["words"].append(symbol)
                            tmp_beam["result_id"].append(id)
                            tmp_beam["word_bs"].append([symbol, bswt * bs_score])
                            tmp_beam["lm"] += lmwt * lmScore
                            tmp_beam["lmState"] = newState

                            # print("frame_id: ", i, ", id: ", id, ", bs: ", bs_score, ", lm: ", tmp_beam["lm"])

                    # 2. compute beam search score
                    tmp_beam["bs"] += bswt * bs_score

                    # 3. record alignment
                    tmp_beam["ali"].append(id)

                    # 4. compute total score with length normalization
                    numFrames = len(tmp_beam["ali"])
                    numWords = len(tmp_beam["words"]) + 1
                    tmp_beam["total"] = tmp_beam["bs"] / numFrames + tmp_beam["lm"] / numWords
                    tmp_beams.append(tmp_beam)

            tmp_beams = sorted(tmp_beams, key=lambda x: x["total"], reverse=True)
            if(len(tmp_beams) == 0):
                continue
            beams_topk = tmp_beams[:beamSize]
        
        # print(sorted(beams_topk, key=lambda x: x["total"], reverse=True)[0])
        # print("bs: ", sorted(beams_topk, key=lambda x: x["total"], reverse=True)[0]['bs'], \
        #         ", lm: ", sorted(beams_topk, key=lambda x: x["total"], reverse=True)[0]['lm'], \
        #         ", total: ", sorted(beams_topk, key=lambda x: x["total"], reverse=True)[0]['total'])
        self.result_id = sorted(beams_topk, key=lambda x: x["total"], reverse=True)[0]['result_id']
        self.word_bs = sorted(beams_topk, key=lambda x: x["total"], reverse=True)[0]['word_bs']
        return 