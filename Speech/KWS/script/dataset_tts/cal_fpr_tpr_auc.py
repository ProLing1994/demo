import argparse
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/common')
from common.utils.python.metrics_tools import *

def cal_fpr_tpr(args):
    pd_csv = pd.read_csv(args.input_scv)

    # 整个测试集（包含：重庆采集数据，深圳采集数据）
    tn = len(pd_csv[pd_csv['label_idx'] == 0][pd_csv['result_idx'] == 0])
    fp = len(pd_csv[pd_csv['label_idx'] == 0][pd_csv['result_idx'] == 1])
    fn = len(pd_csv[pd_csv['label_idx'] == 1][pd_csv['result_idx'] == 0])
    tp = len(pd_csv[pd_csv['label_idx'] == 1][pd_csv['result_idx'] == 1])
    accuracy = get_accuracy(tn, fp, fn, tp)
    tpr = get_tpr(tn, fp, fn, tp)
    fpr = get_fpr(tn, fp, fn, tp)
    print("整个测试集：accuracy: {:.3f}({}/{}), tpr: {:.3f}({}/{}), fpr: {:.3f}({}/{})".format(
        accuracy, tp+tn, tp+fp+tn+fn, tpr,  tp, tp+fn, fpr, fp, fp+tn))

    # 正例测试集划分
    speaker_city_list = []
    for idx, row in pd_csv.iterrows():
        file_name = os.path.basename(row['file'])
        speaker_city = "未知"
        if "RM_KWS_ACTIVATEBWC_activatebwc_" in file_name:
            speaker_city = "重庆" if int(str(file_name).split("_")[-1][1:4]) < 40 else "深圳"
        # print(speaker_city, file_name)
        speaker_city_list.append(speaker_city)
    pd_csv["speaker_city"] = speaker_city_list

    # 重庆测试集
    tn = len(pd_csv[pd_csv["speaker_city"] == "重庆"][pd_csv['label_idx'] == 0][pd_csv['result_idx'] == 0])
    fp = len(pd_csv[pd_csv["speaker_city"] == "重庆"][pd_csv['label_idx'] == 0][pd_csv['result_idx'] == 1])
    fn = len(pd_csv[pd_csv["speaker_city"] == "重庆"][pd_csv['label_idx'] == 1][pd_csv['result_idx'] == 0])
    tp = len(pd_csv[pd_csv["speaker_city"] == "重庆"][pd_csv['label_idx'] == 1][pd_csv['result_idx'] == 1])
    accuracy = get_accuracy(tn, fp, fn, tp)
    tpr = get_tpr(tn, fp, fn, tp)
    fpr = get_fpr(tn, fp, fn, tp)
    print("重庆测试集：accuracy: {:.3f}({}/{}), tpr: {:.3f}({}/{}), fpr: {:.3f}({}/{})".format(
        accuracy, tp+tn, tp+fp+tn+fn, tpr,  tp, tp+fn, fpr, fp, fp+tn))

    # 深圳测试集
    tn = len(pd_csv[pd_csv["speaker_city"] == "深圳"][pd_csv['label_idx'] == 0][pd_csv['result_idx'] == 0])
    fp = len(pd_csv[pd_csv["speaker_city"] == "深圳"][pd_csv['label_idx'] == 0][pd_csv['result_idx'] == 1])
    fn = len(pd_csv[pd_csv["speaker_city"] == "深圳"][pd_csv['label_idx'] == 1][pd_csv['result_idx'] == 0])
    tp = len(pd_csv[pd_csv["speaker_city"] == "深圳"][pd_csv['label_idx'] == 1][pd_csv['result_idx'] == 1])
    accuracy = get_accuracy(tn, fp, fn, tp)
    tpr = get_tpr(tn, fp, fn, tp)
    fpr = get_fpr(tn, fp, fn, tp)
    print("深圳测试集：accuracy: {:.3f}({}/{}), tpr: {:.3f}({}/{}), fpr: {:.3f}({}/{})".format(
        accuracy, tp+tn, tp+fp+tn+fn, tpr,  tp, tp+fn, fpr, fp, fp+tn))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Streamax KWS Infering Engine')
    args = parser.parse_args()
    # args.input_scv = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_1_6_res15_fbankcpu_03222021/dataset_1_6_infer_longterm_validation_augmentation_False_mean.csv"
    # args.input_scv = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_wntts_1_0_res15_fbankcpu_07162021/dataset_1_6_infer_longterm_validation_augmentation_False_mean_epoch_999.csv"
    # args.input_scv = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_tts_1_0_res15_fbankcpu_07162021/dataset_1_6_infer_longterm_validation_augmentation_False_mean_epoch_1000.csv"
    # args.input_scv = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_tts_1_1_res15_fbankcpu_07162021/dataset_1_6_infer_longterm_validation_augmentation_False_mean_epoch_1000.csv"
    # args.input_scv = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_sztts_1_1_res15_fbankcpu_07162021/dataset_1_6_infer_longterm_validation_augmntation_False_mean_epoch_999.csv"

    args.input_scv = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_tts_1_1_10_res15_fbankcpu_07162021/dataset_1_6_infer_longterm_validation_augmentation_False_mean_epoch_500.csv"

    cal_fpr_tpr(args)
