import argparse
import cv2
import os
import pandas as pd
import sys
import shutil
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *


def find_data():
    data_list = get_sub_filepaths_suffix(args.input_dir, suffix='.jpg')

    for data_idx in range(len(data_list)):
        # init 
        input_path = data_list[data_idx]

        img = cv2.imread(input_path)

        # info 
        image_width = img.shape[1]
        image_height = img.shape[0]

        output_path = os.path.join( args.output_dir, os.path.basename(input_path) )
        create_folder( os.path.dirname(output_path) )

        if image_height in args.height_threshold:
            print(input_path, '->', output_path)
            shutil.copy(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # # normal，林旭南提供数据
    # args.input_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/normal/blue/"
    # args.output_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/normal/blue_height_25_30/"
    # # args.input_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/normal/green/"
    # # args.output_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/normal/green_height_25_30/"

    # zg，智观加油站数据 2M
    # args.input_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_2M/blue/"
    # args.output_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_2M/blue_height_25_30/"
    args.input_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_2M/green/"
    args.output_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_2M/green_height_25_30/"
    args.height_threshold = [25, 26, 27, 28, 29, 30]

    find_data()