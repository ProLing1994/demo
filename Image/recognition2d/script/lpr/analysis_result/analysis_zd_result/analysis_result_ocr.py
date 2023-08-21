import argparse
import os
from tkinter.messagebox import NO
import pandas as pd
from prettytable import PrettyTable
import sys
import sklearn.metrics

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from common.common.utils.python.metrics_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_nomal import *


def analysis_total(args):

    print("**********************************")
    print("** {}".format("total"))
    print("**********************************")

    # load csv
    csv_pd = pd.read_csv(args.input_csv_path)

    # init
    true_list = []
    pred_list = []

    for _, row in csv_pd.iterrows():
        true_list.append(1)
        pred_list.append(int(row['res']))

    tn, fp, fn, tp = get_confusion_matrix(true_list, pred_list)
    accuracy = get_accuracy(tn, fp, fn, tp)
    res_dict = {
                    'class': "all", 
                    'acc': "{:.2f}%".format(accuracy * 100), 
                    'num': len(true_list),
                }
    print(res_dict)

    return [res_dict]


def analysis_event(args, event, event_name_list):

    print("**********************************")
    print("** {}".format(event))
    print("**********************************")

    # load csv
    csv_pd = pd.read_csv(args.input_csv_path)
    
    # init 
    res_list = []               # [ {'class': class, 'acc': tpr, 'num': num } ]

    for idx in range(len(event_name_list)):

        analysis_idx = event_name_list[idx]
        analysis_csv_pd = csv_pd[csv_pd[event] == analysis_idx]

        if not len(analysis_csv_pd):
            continue

        # init
        true_list = []
        pred_list = []

        for _, row in analysis_csv_pd.iterrows():
            true_list.append(1)
            pred_list.append(int(row['res']))

        tn, fp, fn, tp = get_confusion_matrix(true_list, pred_list)
        accuracy = get_accuracy(tn, fp, fn, tp)
        res_dict = {
                        'class': "{}_{}".format(event, analysis_idx),
                        'acc': "{:.2f}%".format(accuracy * 100), 
                        'num': len(true_list),
                    }
        
        res_list.append(res_dict)
    
    print(res_list)

    return res_list


def analysis_result(args):
    
    # init 
    res_list = []

    res_list.extend(analysis_total(args))

    # out csv
    error_pd = pd.DataFrame(res_list)
    error_pd.to_csv(args.out_csv_path, index=False, encoding="utf_8_sig")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # #######################################
    # ## zd
    # #######################################
    # args.input_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_pad_20230703/"
    # # ocr_merge_test
    # args.input_csv_path = os.path.join(args.input_dir, 'test/ocr_merge_test_result.csv')
    # args.out_csv_path = os.path.join(args.input_dir, "test/analysis_ocr_merge_test_result.csv")
    # # args.input_csv_path = os.path.join(args.input_dir, 'test_onnx/ocr_merge_test_result.csv')
    # # args.out_csv_path = os.path.join(args.input_dir, "test_onnx/analysis_ocr_merge_test_result.csv")
    # # args.input_csv_path = os.path.join(args.input_dir, 'test_caffe/ocr_merge_test_result.csv')
    # # args.out_csv_path = os.path.join(args.input_dir, "test_caffe/analysis_ocr_merge_test_result.csv")
    # analysis_result(args)

    ######################################
    # chn
    ######################################
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230530_cn"
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707"                    # diffste_248_111600 
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707_original_248"       # original_248_52
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230727_diffste_1_7M"       # diffste_1_7M(diffste_2141_yellow/diffste_2189_green/diffste_3859_blue)
    # args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230727_diffste_1_7M"       # diffste_1_7M(diffste_2141_yellow/diffste_2189_green/diffste_3859_blue)

    ######################################
    # brazil
    ######################################
    args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_brazil_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230818_wdiffste"       # data + diffste(200,000)

    # ocr_merge_test
    # args.input_csv_path = os.path.join(args.input_dir, 'best_accuracy/test_caffe/data_original_248_52_train_result.csv')
    # args.out_csv_path = os.path.join(args.input_dir, "best_accuracy/test_caffe/analysis_data_original_248_52_train_result.csv")
    # args.input_csv_path = os.path.join(args.input_dir, 'best_accuracy/test_caffe/data_original_248_52_val_result.csv')
    # args.out_csv_path = os.path.join(args.input_dir, "best_accuracy/test_caffe/analysis_data_original_248_52_val_result.csv")
    
    # cn
    # args.input_csv_path = os.path.join(args.input_dir, 'best_accuracy/test_caffe/data_sichuan_result.csv')
    # args.out_csv_path = os.path.join(args.input_dir, "best_accuracy/test_caffe/analysis_data_sichuan_result.csv")
    # args.input_csv_path = os.path.join(args.input_dir, 'best_accuracy/test_caffe/data_sichuan_result_no_char.csv')
    # args.out_csv_path = os.path.join(args.input_dir, "best_accuracy/test_caffe/analysis_data_sichuan_result_no_char.csv")
    # args.input_csv_path = os.path.join(args.input_dir, 'inference/test_caffe/data_sichuan_train_result.csv')
    # args.out_csv_path = os.path.join(args.input_dir, "inference/test_caffe/analysis_data_sichuan_train_result.csv")
    # args.input_csv_path = os.path.join(args.input_dir, 'inference/test_caffe/data_sichuan_test_result.csv')
    # args.out_csv_path = os.path.join(args.input_dir, "inference/test_caffe/analysis_data_sichuan_test_result.csv")
    # args.input_csv_path = os.path.join(args.input_dir, 'inference/test_caffe/data_all_no_xianggangaomen_doubleyellow_train_result.csv')
    # args.out_csv_path = os.path.join(args.input_dir, "inference/test_caffe/analysis_data_all_no_xianggangaomen_doubleyellow_train_result.csv")
    # args.input_csv_path = os.path.join(args.input_dir, 'inference/test_caffe/data_all_no_xianggangaomen_doubleyellow_test_result.csv')
    # args.out_csv_path = os.path.join(args.input_dir, "inference/test_caffe/analysis_data_all_no_xianggangaomen_doubleyellow_test_result.csv")

    # brazil
    args.input_csv_path = os.path.join(args.input_dir, 'inference/test_caffe/data_brazil_test_result.csv')
    args.out_csv_path = os.path.join(args.input_dir, "inference/test_caffe/analysis_data_brazil_test_result.csv")

    analysis_result(args)

