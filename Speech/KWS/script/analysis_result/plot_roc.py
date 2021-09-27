import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd 
import sys

from scipy import interp

sys.path.insert(0, '/home/huanyuan/code/demo/common')
from common.utils.python.metrics_tools import get_fpr_tpr, get_auc

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from dataset.kws.dataset_helper import SILENCE_LABEL, UNKNOWN_WORD_LABEL


def plot_roc(fpr, tpr, color, linestyle, label):
    plt.plot(fpr, tpr, color = color,  linewidth=1.0, linestyle=linestyle, marker = 'o', label = label)
    plt.legend(loc=4)
    # plt.xlim([0.0, 0.5])
    # plt.ylim([0.5, 1.01])
    plt.xlim([0.0, 0.05])
    plt.ylim([0.96, 1.01])
    plt.xlabel('Flase Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')


def show_roc_per_class(csv_list, color_list, linestyle_list, label_num, ignore_num):
    csv_path = csv_list[0]
    data_pd = pd.read_csv(csv_path)
        
    mean_fpr = np.linspace(0, 1, 250)
    mean_tpr = 0.0
    for class_idx in range(label_num):
        
        # support for positive label
        if class_idx < ignore_num:
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
        # mean_tpr[0] = 0.0  # 初始处为0

    mean_tpr /= (label_num - ignore_num)
    mean_tpr[-1] = 1.0
    
    plot_roc(mean_fpr, mean_tpr, "c", "--", "mean")
    plt.show()


def show_roc(csv_list, color_list, linestyle_list, name_list, label_num, ignore_num):

    plt.figure()

    for idx in range(len(csv_list)):

        csv_path = csv_list[idx]
        data_pd = pd.read_csv(csv_path)
        mean_fpr = np.linspace(0, 1, 250)
        mean_tpr = 0.0

        for class_idx in range(label_num):
            
            # support for positive label
            if class_idx < ignore_num:
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

        mean_tpr /= (label_num - ignore_num)
        mean_tpr[-1] = 1.0
        auc = get_auc(mean_fpr, mean_tpr)
        plot_roc(mean_fpr, mean_tpr, color_list[idx], linestyle_list[idx], name_list[idx] + " (auc: {:.3f})".format(auc))

    plt.show()


def main():
    # tf_speech_commands
    # csv_list = ["/mnt/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res15_10232020/infer_validation_augmentation_False.csv"]
    # color_list =  ["r", "r", "r", "g", "g", "g", "b", "b", "b", "y", "y", "y"]
    # linestyle_list =  ["-", "--", ":", "-", "--", ":", "-", "--", ":", "-", "--", ":"]
    # label_num = 12
    # ignore_num= 2
    # show_roc_per_class(csv_list, color_list, linestyle_list, label_num, ignore_num)

    # xiaoyu： 3 label
    # csv_list = ["/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui1_0_res15_12032020/infer_longterm_validation_augmentation_False_min.csv"]
    # color_list =  ["r", "g", "b"]
    # linestyle_list =  ["-", "-", "-"]
    # label_num = 3
    # ignore_num= 2
    # show_roc_per_class(csv_list, color_list, linestyle_list, label_num, ignore_num)

    # xiaoyu: 2 label 
    # csv_list = ["/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu7_0_timeshift_spec_on_res15_11192020/infer_longterm_average_validation_augmentation_False.csv"]
    # color_list =  ["r", "g", "b"]
    # linestyle_list =  ["-", "-", "-"]
    # label_num = 2
    # ignore_num= 1
    # show_roc_per_class(csv_list, color_list, linestyle_list, label_num, ignore_num)

    # activatebwc: 2 label 
    # csv_list = ["/mnt/huanyuan/model/model_10_30_25_21/model/kws_activatebwc_1_5_res15_fbankcpu_03222021/dataset_1_4_infer_longterm_validation_augmentation_False_mean.csv"]
    # color_list =  ["r", "g", "b"]
    # linestyle_list =  ["-", "-", "-"]
    # label_num = 2
    # ignore_num= 1
    # show_roc_per_class(csv_list, color_list, linestyle_list, label_num, ignore_num)

    # mutil_label
    # tf_speech_commands
    # csv_list = ["/mnt/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_10212020_le-4/infer_validation_augmentation_False.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_cnn-tpool2_10222020/infer_validation_augmentation_False.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_cnn-one-fstride1_10222020/infer_validation_augmentation_False.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res15_10232020/infer_validation_augmentation_False.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res15-narrow_10232020/infer_validation_augmentation_False.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res8_10232020/infer_validation_augmentation_False.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res8-narrow_10232020/infer_validation_augmentation_False.csv"]
    # color_list = ["r", "r", "r", "g", "g", "b", "b"]
    # linestyle_list = ["-", "--", ":", "-", "--", "-", "--"]
    # name_list = ["cnn-trad-pool2-validation", "cnn-tpool2-validation", "cnn-one-fstride1-validation",
    #             "res15-validation","res15-narrow-validation",
    #             "res8-validation","res8-narrow-validation"]
    # label_num = 12
    # ignore_num= 2
    # show_roc(csv_list, color_list, linestyle_list, name_list, label_num, ignore_num)

    # tf_speech_commands
    # csv_list = ["/mnt/huanyuan/model/model_10_30_25_21/model/kws_with_augmentation_preload_audio_res15_10232020/infer_longterm_validation_augmentation_False_mean.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_speech_1_1_edge-speech-nets_02042021/infer_longterm_validation_augmentation_False_mean.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_speech_1_2_tc-resnet8_02192021/infer_longterm_validation_augmentation_False_mean.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_speech_1_3_tc-resnet14_02192021/infer_longterm_validation_augmentation_False_mean.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_speech_1_4_tc-resnet8-dropout_02192021/infer_longterm_validation_augmentation_False_mean.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_speech_1_5_tc-resnet14-dropout_02192021/infer_longterm_validation_augmentation_False_mean.csv"]
    # color_list = ["y", "r", "g", "b", "g", "b"]
    # linestyle_list = ["-", "-", "-", "-", "--", "--"]
    # name_list = ["res15-validation", "edge-speech-nets", "tc-resnet8", "tc-resnet14", "tc-resnet8-dropout", "tc-resnet14-dropout"]
    # label_num = 12
    # ignore_num= 2
    # show_roc(csv_list, color_list, linestyle_list, name_list, label_num, ignore_num)

    # activatebwc: 2 label 
    csv_list = ["/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_2_5_tc-resnet14-amba_fbankcpu_kd_07162021/dataset_1_6_infer_longterm_validation_augmentation_False_mean.csv",
                "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_2_5_tc-resnet14-amba_fbankcpu_kd_07162021/dataset_1_11_infer_longterm_validation_augmentation_False_mean.csv",
                "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_2_7_tc-resnet14-amba_fbankcpu_kd_09222021/dataset_1_6_infer_longterm_validation_augmentation_False_mean.csv",
                "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english/kws_activatebwc_2_7_tc-resnet14-amba_fbankcpu_kd_09222021/dataset_1_11_infer_longterm_validation_augmentation_False_mean.csv",]
    color_list = ["r", "r", "g", "g"]
    linestyle_list = ["-", "--", "-", "--"]
    name_list = ["activatebwc-tcresnet14(2.5)-dataset(1.6)", "activatebwc-tcresnet14(2.5)-dataset(1.11)", "activatebwc-tcresnet14(2.7)-dataset(1.6)", "activatebwc-tcresnet14(2.7)-dataset(1.11)"]
    label_num = 2
    ignore_num= 1
    show_roc(csv_list, color_list, linestyle_list, name_list, label_num, ignore_num)

    # # xiaoan: 2 label 
    # csv_list = ["/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoan8k_2_5_tc-resnet14-amba_fbankcpu_kd_05152021/dataset_1_7_infer_longterm_validation_augmentation_False_mean.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoan8k_2_5_tc-resnet14-amba_fbankcpu_kd_05152021/dataset_1_8_infer_longterm_validation_augmentation_False_mean.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoan8k_3_1_tc-resnet14-hisi_fbankcpu_kd_05152021/dataset_1_7_infer_longterm_validation_augmentation_False_mean.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoan8k_3_1_tc-resnet14-hisi_fbankcpu_kd_05152021/dataset_1_8_infer_longterm_validation_augmentation_False_mean.csv"]
    # color_list = ["r", "r", "b", "b"]
    # linestyle_list = ["-", "--", "-", "--"]
    # name_list = ["xiaoan8k-res15(2.5)--dataset(1.7)", "xiaoan8k-tcresnet14(2.5)--dataset(1.8)", "xiaoan8k-res15-64(3.1)--dataset(1.7)", "xiaoan8k-tcresnet14-64(3.1)--dataset(1.8)",]
    # label_num = 2
    # ignore_num= 1
    # show_roc(csv_list, color_list, linestyle_list, name_list, label_num, ignore_num)

    # # xiaorui: 2 label
    # csv_list = ["/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui_6_2_tc-resnet14-hisi_fbankcpu_kd_05302021/dataset_1_6_infer_longterm_validation_augmentation_False_mean.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui_6_2_tc-resnet14-hisi_fbankcpu_kd_05302021/dataset_1_7_infer_longterm_validation_augmentation_False_mean.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui_6_3_tc-resnet14-hisi_fbankcpu_kd_05302021/dataset_1_6_infer_longterm_validation_augmentation_False_mean.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui_6_3_tc-resnet14-hisi_fbankcpu_kd_05302021/dataset_1_7_infer_longterm_validation_augmentation_False_mean.csv"]
    # color_list = ["r", "r", "b", "b"]
    # linestyle_list = ["-", ":", "-", ":"]
    # name_list = ["xiaorui16k-tcresnet-hisi-64(6.2)--dataset(1.6)", "xiaorui16k-tcresnet-hisi-64(6.2)--dataset(1.7)", "xiaorui16k-tcresnet-hisi-64(6.3)--dataset(1.6)", "xiaorui16k-tcresnet-hisi-64(6.3)--dataset(1.7)"]
    # label_num = 2
    # ignore_num= 1
    # show_roc(csv_list, color_list, linestyle_list, name_list, label_num, ignore_num)

if __name__ == "__main__":
    main()