import copy
import heapq
import numpy as np
import kenlm


control_command_list = [['start', 'record'], ['stop', 'record'], ['mute', 'audio'], ['unmute', 'audio']]


class Ken_LM:

    def __init__(self, lmPATH):
        self.__model = kenlm.Model(lmPATH)
        self.senlen = 3

    def compute(self, state, word):
        assert isinstance(state, list)
        state.append(word)
        sentence = " ".join(state)
        # 输入需要显式指定<s>起始符，不会默认添加，然后忽略eos终止符，不会为输入的sentence添加默认终止符</s>
        prob = list(self.__model.full_scores(sentence, bos=False, eos=False))[-1][0]
        if len(state) < self.__model.order:
            return state, prob
        else:
            return state[1:], prob


def edit_distance_symbol(word1, word2):
    word1 = "".join(word1.strip().split(' '))
    word2 = "".join(word2.strip().split(' '))
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i-1] == word2[j-1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta,
                           min(dp[i-1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


def edit_distance_pinyin(sentence1, sentence2):
    sentence1 = sentence1.strip().split(' ')
    sentence2 = sentence2.strip().split(' ')
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

def remove_number_english_string(word):
    res_word = ''
    for idx in range(len(word)):
        if word[idx] >= '0' and word[idx] <= '9':
            continue
        else:
            res_word += word[idx]
    return res_word

def match_english_string_verb(key_word, symbol_word):
    key_word_without_number = remove_number_english_string(key_word)
    symbol_word_without_number = remove_number_english_string(symbol_word)

    if len(symbol_word_without_number) < len(key_word_without_number):
        return False
    
    dist = edit_distance_symbol(key_word_without_number, symbol_word_without_number[:len(key_word_without_number)]) 

    if dist == 0:
        return True
    else:
        return False

def match_english_string_noum(key_word, symbol_word):
    key_word_without_number = remove_number_english_string(key_word)
    symbol_word_without_number = remove_number_english_string(symbol_word)

    dist = edit_distance_symbol(key_word_without_number, symbol_word_without_number)
    if dist == 0 or (dist < 2 and len(key_word_without_number) > 4) or (dist < 3 and len(key_word_without_number) > 6):
        return True
    else:
        return match_english_string_verb(key_word, symbol_word)

class Decode(object):
    """ decode python wrapper """

    def __init__(self):
        self.id2word = []
        self.words_scores = []

    def __del__(self):
        pass

    def ctc_decoder(self, input_data):
        # init
        result_id = []
        blank_id = 0
        last_max_id = 0
        frame_num = input_data.shape[0];
        feature_num = input_data.shape[1];

        for idx in range(frame_num):
            max_value = input_data[idx][0]
            max_id = 0

            for idy in range(feature_num):
                if input_data[idx][idy] > max_value:
                    max_value = input_data[idx][idy]
                    max_id = idy
            
            if max_id != blank_id and last_max_id != max_id:
                result_id.append(max_id)
                last_max_id = max_id;
                print("id: ", max_id, ", value: ", max_value)
        return result_id

    def result_id_length(self):
        pass

    def result_id_to_numpy(self):
        pass

    def init_symbol_list(self, asr_dict_path):
        with open(asr_dict_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.id2word.append(line.strip())

    def init_lm_model(self, asr_lm_path):
        self.lm = Ken_LM(asr_lm_path)

    def output_symbol(self, result_id):
        output_symbol = ""
        for idx in range(len(result_id)):
            symbol = self.id2word[result_id[idx]]

            output_symbol += symbol + " "
        return output_symbol

    def output_symbol_english(self, result_id):
        output_symbol = ""
        for idx in range(len(result_id)):
            symbol = self.id2word[result_id[idx]]

            if symbol[0] == '_':
                if idx != 0:
                    output_symbol += " "
                output_symbol += symbol[1:]
            else:
                output_symbol += symbol
        if len(result_id):
            output_symbol += " "
        return output_symbol

    def output_result_string(self, sting_list):
        res_string = ''
        for idx in range(len(sting_list)):
            res_string += sting_list[idx]
            res_string += ' '
        return res_string.strip()

    def match_keywords_english(self, english_symbol_list, kws_list, kws_dict):
        # init
        res_sting_list = []
        matching_lable_list = []            # 容器，记录匹配的 label

        # init matching_state_list，匹配状态容器
        matching_state_list = []            # {'words':[], 'lable':[], 'length':0, 'matched_id':-1}
        for idx in range(len(kws_list)):
            kws_idx = kws_list[idx]
            for idy in range(len(kws_dict[kws_idx])):
                matching_state_dict = {}
                matching_state_dict['words'] = kws_dict[kws_idx][idy].strip().split(" ")
                matching_state_dict['lable'] = kws_idx
                matching_state_dict['length'] = len(kws_dict[kws_idx][idy].strip().split(" "))
                matching_state_dict['matched_id'] = -1
                matching_state_list.append(matching_state_dict)

        # 遍历 english_symbol_list
        for idx in range(len(english_symbol_list)):
            # 遍历 matching_state_list
            for idy in range(len(matching_state_list)):
                # init
                match_bool = False
                words = matching_state_list[idy]['words'];
                lable = matching_state_list[idy]['lable'];
                length = matching_state_list[idy]['length'];
                matched_id = matching_state_list[idy]['matched_id'];

                # 当前策略：
                # 动词：任意匹配 [ing, ed, s]；
                if matched_id + 1 == 0:
                    match_bool = match_english_string_verb(words[matched_id + 1], english_symbol_list[idx])
                # 名词：编辑距离小于 1 或者任意匹配 [ing, ed, s]
                elif matched_id + 1 < length:
                    match_bool = match_english_string_noum(words[matched_id + 1], english_symbol_list[idx])
                else:
                    continue

                if match_bool:
                    # 更新 matched_id
                    matched_id = matched_id + 1
                    matching_state_list[idy]['matched_id'] = matched_id

                    # # 查询匹配成功的字符
                    # print("匹配成功字符：", english_symbol_list[idx], ": ", words[matched_id]);
                    # print("匹配成功长度：", matched_id + 1, "/", length);

                    if matched_id + 1 == length:
                        find_matching_lable_bool = True if lable in matching_lable_list else False

                        # special for label: freeze
                        find_lable_freeze_bool = False
                        output_label_freeze_bool = False
                        if len(self.words_scores):
                            if lable == "freeze":
                                freeze_score = 0
                                freeze_threshold = -1.0
                                feeeze_phoneme_length = 4

                                find_lable_freeze_bool = True
                                if len(self.words_scores) + 1 - feeeze_phoneme_length >= 0:
                                    for idk in range(len(self.words_scores) + 1 - feeeze_phoneme_length):
                                        if (self.words_scores[idk][0] == "_F") and (self.words_scores[idk+1][0]  == "R") and (self.words_scores[idk+2][0]  == "IY1") and (self.words_scores[idk+3][0]  == "Z"):
                                            # 查询字符和得分
                                            print("字符：", self.words_scores[idk][0], ", 得分: ", self.words_scores[idk][1])
                                            print("字符：", self.words_scores[idk+1][0], ", 得分: ", self.words_scores[idk+1][1])
                                            print("字符：", self.words_scores[idk+2][0], ", 得分: ", self.words_scores[idk+2][1])
                                            print("字符：", self.words_scores[idk+3][0], ", 得分: ", self.words_scores[idk+3][1])
                                            freeze_score = self.words_scores[idk][1] + self.words_scores[idk+1][1] + self.words_scores[idk+2][1] + self.words_scores[idk+3][1]

                                            if freeze_score > freeze_threshold:
                                                output_label_freeze_bool = True

                        if not find_matching_lable_bool and (not find_lable_freeze_bool or (find_lable_freeze_bool and output_label_freeze_bool)):
                            matching_lable_list.append(lable)
                            res_sting_list.append(lable)
        return res_sting_list

    def match_kws_english_control_command(self, english_symbol_list):
        # init
        output_control_command_list = []
        output_not_control_command_list = []

        for idx in range(len(english_symbol_list)):
            if english_symbol_list[idx] in control_command_list:
                output_control_command_list.append(english_symbol_list[idx])
            else:
                output_not_control_command_list.append(english_symbol_list[idx])
        return output_control_command_list, output_not_control_command_list

    def ctc_beam_search(self, prob, beamSize, blankID, bswt=1.0, lmwt=0.3, beams_topk=[]):
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
            beams_topk = [{"words": [], "ali":[], "result_id":[0], "words_scores":[], "lmState":["<s>"], "bs":0, "lm":init_lm_score, "total":0}]
            # beams_topk = [{"words": [], "ali":[], "result_id":[0], "words_scores":[],"lmState":["<s>"], "bs":0, "lm":0.0, "total":0}]

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
                        # if len(tmp_beam["ali"]) == 1 or id != tmp_beam["result_id"][-1]:
                            symbol = self.id2word[id]
                            newState, lmScore = self.lm.compute(state=tmp_beam["lmState"], word=symbol)
                            tmp_beam["words"].append(symbol)
                            tmp_beam["result_id"].append(id)
                            tmp_beam["words_scores"].append([symbol, bswt * bs_score])
                            tmp_beam["lm"] += lmwt * lmScore
                            tmp_beam["lmState"] = newState

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
        
        result_id = sorted(beams_topk, key=lambda x: x["total"], reverse=True)[0]['result_id'][1:]
        self.words_scores = sorted(beams_topk, key=lambda x: x["total"], reverse=True)[0]['words_scores']
        return result_id
