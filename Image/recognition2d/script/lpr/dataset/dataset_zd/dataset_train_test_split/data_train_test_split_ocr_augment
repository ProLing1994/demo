import argparse
import os
import pandas as pd
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.recognition2d.script.lpr.dataset.dataset_zd.dataset_train_test_split.data_train_test_split_ocr import split


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="uae_20220804_0809") 
    parser.add_argument('--ocr_name', type=str, default="plate_zd_mask_202306") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/")  
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.ocr_name, args.date_name)

    print("data train test split augment.")
    print("date_name: {}".format(args.date_name))
    print("ocr_name: {}".format(args.ocr_name))
    print("input_dir: {}".format(args.input_dir))

    args.input_csv_path = os.path.join(args.input_dir, '{}_augument.csv'.format(args.date_name))
    args.to_trainval_file = os.path.join(args.input_dir, "ImageSets_aug/ImageSets/Main/trainval.txt")
    args.to_train_file = os.path.join(args.input_dir, "ImageSets_aug/ImageSets/Main/train.txt")
    args.to_val_file = os.path.join(args.input_dir, "ImageSets_aug/ImageSets/Main/val.txt")
    args.to_test_file = os.path.join(args.input_dir, "ImageSets_aug/ImageSets/Main/test.txt")

    split(args)
