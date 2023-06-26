import argparse
import cv2
import importlib
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from script.lpr.dataset.dataset_augment.data_augment import *
from script.lpr.dataset.dataset_zd.dataset_mask.gen_ocr_img import *


def gen_ocr_img_augument(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.lpr.dataset.dataset_zd.dataset_dict.' + args.data_dict_name) 

    # mkdir
    create_folder(args.output_img_dir)
    create_folder(args.output_error_data_dir)

    # interval
    args.interval_img = load_interval_img(args)   

    # pd
    args.data_pd = pd.read_csv(args.input_csv_path)

    # gen_img
    gen_img(args, dataset_dict)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="uae_20220804_0809") 
    parser.add_argument('--seg_name', type=str, default="seg_zd_202306") 
    parser.add_argument('--ocr_name', type=str, default="plate_zd_mask_202306") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/") 
    args = parser.parse_args()

    args.seg_dir = os.path.join(args.output_dir, args.seg_name, args.date_name)
    args.output_dir = os.path.join(args.output_dir, args.ocr_name, args.date_name)

    print("gen ocr img augment.")
    print("date_name: {}".format(args.date_name))
    print("seg_name: {}".format(args.seg_name))
    print("ocr_name: {}".format(args.ocr_name))
    print("output_dir: {}".format(args.output_dir))

    args.input_csv_path = os.path.join(args.seg_dir, 'city_label', args.date_name + '_augument.csv')
    args.output_img_dir = os.path.join(args.output_dir, "Images_aug")
    args.output_error_data_dir = os.path.join(args.output_dir, "error_data_aug")
    args.output_csv_path = os.path.join(args.output_dir, args.date_name + '_augument.csv')

    args.data_dict_name = "dataset_zd_dict_city"
    args.interval_img_path = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/type/20220826_000057_none_none_none_5#7739.jpg"
    args.interval_json_path = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/type/20220826_000057_none_none_none_5#7739.json"

    args.hisi_min_size_threh = 32
    args.kind_min_size_threh = 6
    args.num_min_size_threh = 10
    args.random_expand_ratio = 0.08

    gen_ocr_img_augument(args)