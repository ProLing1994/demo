import argparse
import cv2
import numpy as np
import os
import sys 
import shutil
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *


def move_date(args):
    
    data_list = np.array(os.listdir(args.data_dir))
    data_list = data_list[[jpg.endswith(args.suffix) for jpg in data_list]]
    data_list.sort()

    for idx in tqdm(range(len(data_list))):
        # jpg
        data_name = data_list[idx]
        data_path = os.path.join(args.data_dir, data_name)

        refer_data_name = data_name.replace(args.suffix, ".jpg")
        refer_data_path = os.path.join(args.refer_jpg_dir, refer_data_name)

        output_date_path = os.path.join(args.refer_jpg_dir, data_name)
        if os.path.exists(refer_data_path):
            print(data_path)
            shutil.move(data_path, output_date_path)


def remove_date(args):

    data_list = np.array(os.listdir(args.data_dir))
    data_list = data_list[[jpg.endswith(args.suffix) for jpg in data_list]]
    data_list.sort()

    for idx in tqdm(range(len(data_list))):
        # jpg
        data_name = data_list[idx]
        data_path = os.path.join(args.data_dir, data_name)

        refer_data_name = data_name.replace(args.suffix, ".jpg")
        refer_data_path = os.path.join(args.refer_jpg_dir, refer_data_name)

        if os.path.exists(refer_data_path):
            print(data_path)
            os.remove(data_path)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_C28_detection/america_new/temp/fangche") 
    args = parser.parse_args()

    args.refer_jpg_dir = os.path.join(args.input_dir, "JPEGImages_unknow/")

    args.data_dir = os.path.join(args.input_dir, "JPEGImages/")
    args.suffix = ".jpg"
    move_date(args)

    args.data_dir = os.path.join(args.input_dir, "xml/")
    args.suffix = ".xml"
    move_date(args)

    args.data_dir = os.path.join(args.input_dir, "json_v0/")
    args.suffix = ".json"
    move_date(args)

    # args.data_dir = os.path.join(args.input_dir, "Annotations_Car/")
    # args.suffix = ".xml"
    # remove_date(args)

    