import argparse
import cv2
from collections import OrderedDict
import importlib
import numpy as np
import os
from prettytable import PrettyTable
import sys 
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d')
from script.lpr.analysis_result.analysis_zd_result.metrics import eval_metrics


def analysis_result(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module(args.seg_dict_name) 
    
    # img list
    img_list = []
    city_mask_list = []
    color_mask_list = []

    with open(args.input_test_file_path) as f:
        for line in f:
            img_list.append(line.strip().split(".jpg ")[0] + '.jpg')
            city_mask_list.append(line.split(".jpg ")[1].split(".png")[0] + '.png')
            color_mask_list.append(line.split(".jpg ")[1].split(".png ")[1].strip())

    # load mask
    gt_seg_maps = []
    pred_seg_maps = []

    for idx in tqdm(range(len(city_mask_list))):

        gt_seg_path = city_mask_list[idx]
        gt_seg_name = os.path.basename(gt_seg_path)
        pred_seg_path = os.path.join(args.mask_pred_dir, gt_seg_name)

        gt_seg = cv2.imread(gt_seg_path)[:, :, 0]
        gt_seg = cv2.resize(gt_seg, (128, 64))
        pred_seg = cv2.imread(pred_seg_path)[:, :, 0]

        gt_seg_maps.append(gt_seg)
        pred_seg_maps.append(pred_seg)
    
    gt_seg_maps = np.array(gt_seg_maps)
    pred_seg_maps = np.array(pred_seg_maps)

    # test a list of files
    num_classes = len(dataset_dict.class_seg_label)
    ignore_index = 255

    ret_metrics = eval_metrics(
        pred_seg_maps,
        gt_seg_maps,
        num_classes,
        ignore_index,
        args.metric_list,
        label_map=dict(),
        reduce_zero_label=False)

    # summary table
    ret_metrics_summary = OrderedDict({
        ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })

    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })

    ret_metrics_class.update({'Class': dataset_dict.class_seg_label})
    ret_metrics_class.move_to_end('Class', last=False)

    # for logger
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            summary_table_data.add_column(key, [val])
        else:
            summary_table_data.add_column('m' + key, [val])

    print(class_table_data)
    print(summary_table_data)

    with open(args.mask_res_class_table_csv_path, "w") as f:
        for key, val in ret_metrics_class.items():
            f.write(key)
            f.write(",")
            val = [str(idx) for idx in val]
            f.write(",".join(val))
            f.write("\n")

    with open(args.mask_res_summary_table_csv_path, "w") as f:
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                f.write(key)
                f.write(",")
                f.write(str(val))
                f.write("\n")
            else:
                f.write('m' + key)
                f.write(",")
                f.write(str(val))
                f.write("\n")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ###############################################
    # dataset_zd_dict_city
    ###############################################

    args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230703/"
    args.seg_dict_name = "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_city"

    args.input_dir = "/yuanhuan/data/image/RM_ANPR/training/seg_zd_202307/"
    args.dataset_name = "ImageSetsLabelNoAug/city_color_label"
    args.mode = "test"
    args.input_test_file_path = os.path.join(args.input_dir, args.dataset_name, "ImageSets/Main/{}.txt".format(args.mode))

    args.mask_pred_dir = os.path.join(args.output_dir, '{}/{}/mask_city'.format(args.mode, args.dataset_name))
    args.mask_res_class_table_csv_path = os.path.join(args.output_dir, '{}/{}/mask_res_class_table.csv'.format(args.mode, args.dataset_name))
    args.mask_res_summary_table_csv_path = os.path.join(args.output_dir, '{}/{}/mask_res_summary_table.csv'.format(args.mode, args.dataset_name))
    args.metric_list = ['mIoU', 'mDice', 'mFscore']

    analysis_result(args)

