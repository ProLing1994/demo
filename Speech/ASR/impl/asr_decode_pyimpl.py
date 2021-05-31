import copy
import heapq
import numpy as np
import kenlm


kws_list = [['start', 'record'], ['stop', 'record'], ['mute', 'audio'], ['unmute', 'audio'],
            ['shot', 'fire'], ['freeze'], ['drop', 'gun'], ['keep', 'hand'], ['put', 'hand'], ['down', 'ground']]

control_command_list = [['start', 'record'], [
    'stop', 'record'], ['mute', 'audio'], ['unmute', 'audio']]


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


class Decode(object):
    """ decode python wrapper """

    def __init__(self):
        self.id2word = []

    def __del__(self):
        pass

    def ctc_decoder(self, input_data):
        pass

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

    def match_kws_english(self, string_list):
        # init
        match_list = []
        output_list = []

        # 使用堆的方法进行遍历
        # 遍历字符串
        for idx in range(len(string_list)):
            string_idx = string_list[idx]

            # 遍历模板
            bool_find_kws = False
            for idy in range(len(kws_list)):
                kws_idx = kws_list[idy]

                # 遍历模板成员
                for idz in range(len(kws_idx)):
                    # To do：匹配策略，即使更新
                    # 匹配策略，任意匹配 [ing, ed, s]
                    if kws_idx[idz] in string_idx and kws_idx[idz] == string_idx[:len(kws_idx[idz])]:
                        # 若遍历至模板成员最后一个成员，遍历堆中结果，判断模板成员是否均存在
                        if idz == len(kws_idx) - 1:
                            if len(match_list) >= len(kws_idx) - 1:
                                # 针对长度为 1 的匹配词
                                bool_find_kws = True if len(kws_idx) - 1 == 0 else False

                                # 针对长度大于 1 的匹配词，逆序匹配
                                for idf in range(len(kws_idx) - 1):
                                    bool_find_kws = True if match_list[-1 -
                                                                       idf] == kws_idx[-2 - idf] else False
                        else:
                            # 建堆, 添加到匹配池中
                            match_list.append(kws_idx[idz])

                        if bool_find_kws:
                            output_list.append(kws_idx)

                            # 清除匹配池
                            for idf in range(len(kws_idx) - 1):
                                match_list.pop()

                if bool_find_kws:
                    break

        return output_list

    def match_kws_english_control_command(self, string_list):
        # init
        output_control_command_list = []
        output_not_control_command_list = []

        for idx in range(len(string_list)):
            if string_list[idx] in control_command_list:
                output_control_command_list.append(string_list[idx])
            else:
                output_not_control_command_list.append(string_list[idx])
        return output_control_command_list, output_not_control_command_list

    def ctc_beam_search(self, prob, beamSize, blankID, bswt=1.0, lmwt=0.3, beams_topk=[]):
        # 取对数，np.log，用于和语言模型得分相加
        prob = np.log(prob)

        # init
        frame_num = prob.shape[0]

        # initilize LM score (一定要用一个非0的值来初始化，我根据经验，使用未知词的概率来初始化。)
        _, init_lm_score = self.lm.compute(state=[], word="UNK")
        init_lm_score *= lmwt
        if(len(beams_topk) == 0):
            beams_topk = [{"words": [""], "ali":[0], "bs":0, "lm":init_lm_score, "total":0, "lmState":["<s>"]}]

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

                    tmp_beam = copy.deepcopy(preResult)

                    # 1. compute LM score
                    if id != blankID:  # ignore blank
                        # ignore continuous repeated symbols
                        if len(tmp_beam["ali"]) == 1 or id != tmp_beam["ali"][-1]:
                            symbol = self.id2word[id]
                            newState, lmScore = self.lm.compute(state=tmp_beam["lmState"], word=symbol)
                            tmp_beam["words"].append(symbol)
                            tmp_beam["lm"] += lmwt * lmScore
                            tmp_beam["lmState"] = newState
                            # print("symbol: ", symbol, ", lmScore: ", lmScore)

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
            # print()
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
        return sorted(beams_topk, key=lambda x: x["total"], reverse=True)
