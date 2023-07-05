import argparse
import importlib
import os
import pandas as pd
from prettytable import PrettyTable
import sys
import sklearn.metrics

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from common.common.utils.python.metrics_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d')


def analysis_event(args, event, dataset_dict, event_name_list):

    # load csv
    csv_pd = pd.read_csv(args.input_csv_path)

    # init 
    true_list = []
    pred_list = []

    for _, row in csv_pd.iterrows():
        true_list.append(dataset_dict.name_2_id_dict[row[event]])
        pred_list.append(dataset_dict.name_2_id_dict[row['{}_res'.format(event)]])

    print("**********************************")
    print("** {}".format(event))
    print("**********************************")
    print(sklearn.metrics.confusion_matrix(true_list, pred_list))

    res_dict = {
                    'class': [],
                    'accuracy': [],
                    'tpr': [],
                    'ppv': [],
                    'fpr': [],
                }

    for idx in range(len(event_name_list)):
        
        analysis_idx = event_name_list[idx]

        if analysis_idx not in dataset_dict.name_2_id_dict:
            print("Unknown: {}".format(analysis_idx))
            continue

        true_list_idx = [1 if res == dataset_dict.name_2_id_dict[analysis_idx] else 0 for res in true_list]
        pred_list_idx = [1 if res == dataset_dict.name_2_id_dict[analysis_idx] else 0 for res in pred_list]

        tn, fp, fn, tp = get_confusion_matrix(true_list_idx, pred_list_idx)
        accuracy = get_accuracy(tn, fp, fn, tp)
        tpr = get_tpr(tn, fp, fn, tp)
        fpr = get_fpr(tn, fp, fn, tp)
        ppv = get_ppv(tn, fp, fn, tp)

        res_dict['class'].append(analysis_idx)
        res_dict['accuracy'].append("{:.2f}".format(accuracy))
        res_dict['tpr'].append("{:.2f}".format(tpr))
        res_dict['ppv'].append("{:.2f}".format(ppv))
        res_dict['fpr'].append("{:.2f}".format(fpr))

    tb = PrettyTable()
    for key, val in res_dict.items():
        tb.add_column(key, val)
    print(tb)

    res_table_csv_path = os.path.join(args.output_event_dir, "res_table_{}.csv".format(event))
    with open(res_table_csv_path, "w") as f:
        for key, val in res_dict.items():
            f.write(key)
            f.write(",")
            val = [str(idx) for idx in val]
            f.write(",".join(val))
            f.write("\n")


def analysis_result(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module(args.seg_dict_name) 

    for key in dataset_dict.class_seg_label_group_2_name_map.keys():
        analysis_event(args, key, dataset_dict, dataset_dict.class_seg_label_group_2_name_map[key])


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ###############################################
    # dataset_zd_dict_city & dataset_zd_dict_color
    ###############################################

    args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230703/"

    args.seg_dict_name = "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_city"
    # args.seg_dict_name = "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_color"
    args.dataset_name = "ImageSetsLabelNoAug/city_color_label"
    args.mode = "test"
    args.input_csv_path = os.path.join(args.output_dir, '{}/{}/result.csv'.format(args.mode, args.dataset_name))
    args.output_event_dir = os.path.join(args.output_dir, '{}/{}'.format(args.mode, args.dataset_name))

    analysis_result(args)