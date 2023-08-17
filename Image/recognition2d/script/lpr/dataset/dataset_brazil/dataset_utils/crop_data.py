import argparse
import cv2
import os
import pandas as pd
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def crop_data(args):

    # pd

    # img list 
    img_list = get_sub_filepaths_suffix(args.input_dir, ".jpg")
    img_list.sort()
    print("data len: ", len(img_list))

    # file_list
    for idx in tqdm(range(len(img_list))):

        img_path = img_list[idx]

        # img
        img = cv2.imread(img_path)

        # 获取图像高度和宽度
        height, width = img.shape[:2]

        # assert width == 256 and height == 85

        # 计算裁剪区域的起始位置
        start_h = int((height - 85) / 2)
        end_h = start_h + 85

        # 裁剪图像
        cropped_image = img[start_h:end_h, :]

        # 保存
        cv2.imwrite(img_path, cropped_image)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="original_563_new_style") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/DIFFSTE/")  
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.date_name)

    print("crop data.")
    print("date_name: {}".format(args.date_name))
    print("input_dir: {}".format(args.input_dir))

    crop_data(args)