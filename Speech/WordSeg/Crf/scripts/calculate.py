import codecs
import sys

import sklearn
import sklearn.metrics

label_file  = codecs.open(sys.argv[1], 'r', 'utf-8')
pred_file = codecs.open(sys.argv[2], 'r', 'utf-8')
# label_file  = codecs.open("/home/huanyuan/code/third_code/tts/TTS_Course/02_front_end/2-1_word_seg/data/demo/test.lable", 'r', 'utf-8')
# pred_file = codecs.open("/home/huanyuan/code/third_code/tts/TTS_Course/02_front_end/2-1_word_seg/data/demo/test.out", 'r', 'utf-8')

labels  = ["S", "B", "M", "E"]
target_names = ["S", "B", "M", "E"]
y_true = []
y_pred = [] 

for line in label_file.readlines():
    word_list = line.strip().split()
    if len(word_list) == 2:
        y_true.append(word_list[1])

for line in pred_file.readlines():
    word_list = line.strip().split()
    if len(word_list) == 2:
        y_pred.append(word_list[1])

label_file.close()
pred_file.close()

assert len(y_true) == len(y_pred)
print(" Total data num: {}".format(len(y_pred)))
print("\n Confusion Matrix: ")
print(sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels))
print("\n Classification Report: ")
print(sklearn.metrics.classification_report(y_true, y_pred, labels=labels , target_names=target_names))