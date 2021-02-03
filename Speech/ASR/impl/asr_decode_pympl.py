import numpy as np


def edit_distance_symbol(word1, word2):
    word1 = "".join(word1.strip().split(' '))
    word2 = "".join(word2.strip().split(' '))
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1,len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i    
    for j in range(len2 + 1):
        dp[0][j] = j
     
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i-1] == word2[j-1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i-1][j] + 1, dp[i][j - 1] + 1))
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
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i-1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


class Decode(object):
    """ decode python wrapper """ 
    def __init__(self):
        pass

    def __del__(self):
        pass

    def ctc_decoder(self, input_data):
        pass 

    def result_id_length(self):
        pass

    def result_id_to_numpy(self):
        pass

    def match_kws_english(self, string_list, kws_list):
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
                                    bool_find_kws = True if match_list[-1 - idf] == kws_idx[-2 - idf] else False    
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