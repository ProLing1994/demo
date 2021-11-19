import copy
import heapq
import numpy as np
import kenlm
import re


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


def get_ouststr(pred):
    object_list=['空调','天窗','收音机','地图','前排窗户','蓝牙','后排窗户','音量','温度']
    operation_list=['打开','关闭','连接','断开','调节']
    num_dict={'一':'1','二':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9'}
    num_list=['一','二','三','四','五','六','七','八','九','十','百']
    out_str=''
    if(pred==None):
        return ''
    pred=''.join(pred)
    for object in object_list:
        if(object in pred):
            out_str='control-'+object
            break
    for operation in operation_list:
        if(operation in pred):
            out_str=out_str+'-'+operation
            break
    start_pos=re.search('到',pred)
    if(start_pos==None):
        out_unit=out_str.split('-')
        if(len(out_unit)!=3):
            return ''
        else:
            operation=out_unit[-1]
            if(operation in['打开','连接']):
                out_str=out_str+'-'+'100'
            elif(operation in['关闭','断开']):
                out_str=out_str+'-'+'0'
            else:
                return ''
    else:
        value=''
        start_pos=start_pos.span()[0]
        if(pred[start_pos+1:start_pos+4]=='百分之'):
            start_pos+=4
        else:
            start_pos+=1
        while start_pos<len(pred) and pred[start_pos] in num_list :
            if(pred[start_pos]=='十'):
               if(len(value)==0):
                  value='1'
               elif(start_pos==len(pred) - 1 or pred[start_pos+1] not in num_list):
                  value+='0'
            elif(pred[start_pos]=='百'):
                if(len(value)==0):
                    value='100'
                else:
                    value+='00'
            else:
                value+=num_dict[pred[start_pos]]
            start_pos+=1
        out_str+='-'+value
    return out_str

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

    def output_symbol(self):
        output_symbol = ""
        for idx in range(len(self.result_id)):
            symbol = self.id2word[self.result_id[idx]]
            output_symbol += symbol + " "
        return output_symbol

    def output_symbol_list(self):
        output_symbol_list = []
        for idx in range(len(self.result_id)):
            symbol = self.id2word[self.result_id[idx]]
            output_symbol_list.append(symbol)
        return output_symbol_list

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


