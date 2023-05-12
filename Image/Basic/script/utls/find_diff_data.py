import argparse
import cv2
import numpy as np
import os
import sys 
import shutil
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *


def find_diff_data(args):
    
    # create_folder
    create_folder(args.output_dir)

    data_list = np.array(os.listdir(args.data_dir))
    data_list = data_list[[jpg.endswith(args.suffix) for jpg in data_list]]
    data_list.sort()

    for idx in tqdm(range(len(data_list))):
        data_name = data_list[idx]
        data_path = os.path.join(args.data_dir, data_name)

        # jpg
        refer_data_name = data_name.replace(args.suffix, ".jpg")
        refer_data_path = os.path.join(args.refer_jpg_dir, refer_data_name)

        if not os.path.exists(refer_data_path):
            output_date_path = os.path.join(args.output_dir, data_name)
            print(data_path)
            shutil.copy(data_path, output_date_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_C27_detection/zd_car_plate/_/car_folder/1115/") 
    # parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_C27_detection/zd_car_plate/_/car_folder/0909/") 
    # parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_C27_detection/zd_car_plate/_/car_folder/0901/") 
    # parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_C27_detection/zd_car_plate/_/check_night/") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/temp/zd_plate/") 
    args = parser.parse_args()

    args.refer_jpg_dir = os.path.join("/yuanhuan/data/image/RM_C27_detection/zd_c27_20200209_20201125/", "JPEGImages/")
    
    args.data_dir = os.path.join(args.input_dir, "JPEGImages/")
    args.suffix = ".jpg"
    args.output_dir = os.path.join("/yuanhuan/data/image/RM_C27_detection/zd_c27_new_night/", "JPEGImages/")
    find_diff_data(args)

    args.data_dir = os.path.join(args.input_dir, "XML/")
    args.suffix = ".xml"
    args.output_dir = os.path.join("/yuanhuan/data/image/RM_C27_detection/zd_c27_new_night/", "XML/")
    find_diff_data(args)

    