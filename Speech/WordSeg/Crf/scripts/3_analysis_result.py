import codecs
import sys

import sklearn
import sklearn.metrics

# ###########
# # 方式一：按照字级别计算统计结果
# ########### 

# # label_file  = codecs.open(sys.argv[1], 'r', 'utf-8')
# # pred_file = codecs.open(sys.argv[2], 'r', 'utf-8')
# label_file  = codecs.open("/home/huanyuan/code/third_code/tts/TTS_Course/02_front_end/2-1_word_seg/data/demo/test.lable", 'r', 'utf-8')
# pred_file = codecs.open("/home/huanyuan/code/third_code/tts/TTS_Course/02_front_end/2-1_word_seg/data/demo/test.out", 'r', 'utf-8')

# labels  = ["S", "B", "M", "E"]
# target_names = ["S", "B", "M", "E"]
# y_true = []
# y_pred = [] 

# for line in label_file.readlines():
#     word_list = line.strip().split()
#     if len(word_list) == 2:
#         y_true.append(word_list[1])

# for line in pred_file.readlines():
#     word_list = line.strip().split()
#     if len(word_list) == 2:
#         y_pred.append(word_list[1])

# label_file.close()
# pred_file.close()

# assert len(y_true) == len(y_pred)
# print(" Total data num: {}".format(len(y_pred)))
# print("\n Confusion Matrix: ")
# print(sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels))
# print("\n Classification Report: ")
# print(sklearn.metrics.classification_report(y_true, y_pred, labels=labels , target_names=target_names))


###########
# 方式二：按照分词级别计算统计结果
########### 

import re

label_file  = codecs.open(sys.argv[1], 'r', 'utf-8')
pred_file = codecs.open(sys.argv[2], 'r', 'utf-8')
# label_file  = codecs.open("/home/huanyuan/code/third_code/tts/TTS_Course/02_front_end/2-1_word_seg/data/demo/test.lable", 'r', 'utf-8')
# pred_file = codecs.open("/home/huanyuan/code/third_code/tts/TTS_Course/02_front_end/2-1_word_seg/data/demo/test.out", 'r', 'utf-8')

def to_region(segmentation: str) -> list:
    """
    将分词结果转换为区间
    :param segmentation: 商品 和 服务
    :return: [(0, 2), (2, 3), (3, 5)]
    """
    region = []
    start = 0
    for word in re.compile("\\s+").split(segmentation.strip()):
        end = start + len(word)
        region.append((start, end))
        start = end
    return region

def prf(gold: str, pred: str, dic) -> tuple:
    """
    计算P、R、F1
    :param gold: 标准答案文件，比如“商品 和 服务”
    :param pred: 分词结果文件，比如“商品 和服 务”
    :param dic: 词典
    :return: (P, R, F1, OOV_R, IV_R)
    """
    A_size, B_size, A_cap_B_size, OOV, IV, OOV_R, IV_R = 0, 0, 0, 0, 0, 0, 0
    A, B = set(to_region(gold)), set(to_region(pred))
    A_size += len(A)
    B_size += len(B)
    A_cap_B_size += len(A & B)
    text = re.sub("\\s+", "", gold)

    for (start, end) in A:
        word = text[start: end]
        if word in dic:
            IV += 1
        else:
            OOV += 1

    for (start, end) in A & B:
        word = text[start: end]
        if word in dic:
            IV_R += 1
        else:
            OOV_R += 1
    p = A_cap_B_size / B_size * 100
    r = A_cap_B_size / A_size * 100
    f1 = 2 * p * r / (p + r)
    oov_r = OOV_R / OOV * 100 if OOV != 0 else 0
    iv_r = IV_R / IV * 100 if IV != 0 else 0
    return p, r, f1, oov_r, iv_r

dic = ['结婚', '尚未', '的', '和', '青年', '都', '应该', '好好考虑', '自己',  '人生', '大事']
gold = '结婚 的 和 尚未 结婚 的 都 应该 好好 考虑 一下 人生 大事'
pred = '结婚 的 和尚 未结婚 的 都 应该 好好考虑 一下 人生大事'
print("Precision:%.2f Recall:%.2f F1:%.2f OOV-R:%.2f IV-R:%.2f" % prf(gold, pred, dic))

def to_string(file):

    res_string = ""
    for line in file.readlines():
        word_list = line.strip().split()
        if len(word_list) == 2:
            if word_list[1] == "S":
                res_string += " " + word_list[0] + " " 
            elif  word_list[1] == "B":
                res_string += " " + word_list[0]
            elif  word_list[1] == "M":
                res_string += word_list[0]
            elif  word_list[1] == "E":
                res_string += word_list[0] + " "
    
    res_string = " ".join(res_string.split("  "))
    return res_string

string_y_true = to_string(label_file)
string_y_pred = to_string(pred_file)
dic = list(set(string_y_true.split(" ")))
print("Precision:%.2f Recall:%.2f F1:%.2f OOV-R:%.2f IV-R:%.2f" % prf(string_y_true, string_y_pred, dic))