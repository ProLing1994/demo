import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd 
import sys

from scipy import interp

sys.path.insert(0, '/home/huanyuan/code/demo/common')
from common.utils.python.cal_roc_auc import get_fpr_tpr

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from dataset.kws.daataset_helper import SILENCE_INDEX, UNKNOWN_WORD_INDEX


def plot_roc(fpr, tpr, color, linestyle, label):
    plt.plot(fpr, tpr, color = color,  linewidth=1.0, linestyle=linestyle, marker = 'o', label = label)
    plt.legend(loc=4)
    plt.xlim([0, 0.1])
    plt.ylim([0.7, 1])
    plt.xlabel('Flase Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')


def show_roc_per_class(csv_list, color_list, linestyle_list, num_classes):
    csv_path = csv_list[0]
    data_pd = pd.read_csv(csv_path)
        
    mean_fpr = np.linspace(0, 1, 250)
    mean_tpr = 0.0
    for class_idx in range(num_classes):
        
        # support for positive label
        if class_idx == SILENCE_INDEX or class_idx == UNKNOWN_WORD_INDEX:
            continue

        # load labels/scores
        labels = []
        scores = []
        for  _, row in data_pd.iterrows():
            # print(row['label_idx'], row['prob_{}'.format(row['label_idx'])])
            labels.append(1 if row['label_idx'] == class_idx else 0)
            scores.append(row['prob_{}'.format(class_idx)])
        
        fpr, tpr, thresholds = get_fpr_tpr(labels, scores)
        plot_roc(mean_fpr, np.interp(mean_fpr, fpr, tpr), color_list[class_idx], linestyle_list[class_idx], class_idx)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0  # 初始处为0
    
    mean_tpr /= (num_classes - 2)
    mean_tpr[-1] = 1.0
    
    plot_roc(mean_fpr, mean_tpr, "c", "--", "mean")
    plt.show()


def show_roc(csv_list, color_list, linestyle_list, name_list, num_classes):

    plt.figure()

    for idx in range(len(csv_list)):

        csv_path = csv_list[idx]
        data_pd = pd.read_csv(csv_path)
        mean_fpr = np.linspace(0, 1, 250)
        mean_tpr = 0.0

        for class_idx in range(num_classes):
            
            # support for positive label
            if class_idx == SILENCE_INDEX or class_idx == UNKNOWN_WORD_INDEX:
                continue

            # load labels/scores
            labels = []
            scores = []
            for  _, row in data_pd.iterrows():
                # print(row['label_idx'], row['prob_{}'.format(row['label_idx'])])
                labels.append(1 if row['label_idx'] == class_idx else 0)
                scores.append(row['prob_{}'.format(class_idx)])
            
            fpr, tpr, thresholds = get_fpr_tpr(labels, scores)
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0  # 初始处为0
        
        mean_tpr /= (num_classes - 2)
        mean_tpr[-1] = 1.0
        plot_roc(mean_fpr, mean_tpr, color_list[idx], linestyle_list[idx], name_list[idx])

    plt.show()


def main():
    csv_list = ["/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res15_10232020/infer_validation_augmentation_False.csv"]
    color_list =  ["r", "r", "r", "g", "g", "g", "b", "b", "b", "y", "y", "y"]
    linestyle_list =  ["-", "--", ":", "-", "--", ":", "-", "--", ":", "-", "--", ":"]
    num_classes = 12
    show_roc_per_class(csv_list, color_list, linestyle_list, num_classes)

    # csv_list = ["/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_10212020_le-4/infer_validation_augmentation_False.csv",
    #             "/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_cnn-tpool2_10222020/infer_validation_augmentation_False.csv",
    #             "/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_cnn-one-fstride1_10222020/infer_validation_augmentation_False.csv",
    #             "/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res15_10232020/infer_validation_augmentation_False.csv",
    #             "/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res15-narrow_10232020/infer_validation_augmentation_False.csv",
    #             "/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res8_10232020/infer_validation_augmentation_False.csv",
    #             "/home/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res8-narrow_10232020/infer_validation_augmentation_False.csv"]
    # color_list = ["r", "r", "r", "g", "g", "b", "b"]
    # linestyle_list = ["-", "--", ":", "-", "--", "-", "--"]
    # name_list = ["cnn-trad-pool2-validation", "cnn-tpool2-validation", "cnn-one-fstride1-validation",
    #             "res15-validation","res15-narrow-validation",
    #             "res8-validation","res8-narrow-validation"]
    # num_classes = 12
    # show_roc(csv_list, color_list, linestyle_list, name_list, num_classes)


if __name__ == "__main__":
    main()