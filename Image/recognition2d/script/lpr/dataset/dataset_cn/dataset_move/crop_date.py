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
    
    # init
    csv_list = []           # [{"img_path": "", "json_path": "", "roi_img_path": "", "id": "", "name": "", "roi": "", "color": "", "column": "", "num": "", "split_type": ""}]
    file_list = []          # ["num"]

    # pd
    data_pd = pd.read_csv(args.input_csv_path)

    # file_list
    for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        img_path  = row['roi_img_path']

        # img
        img = cv2.imread(img_path)

        # 获取图像高度和宽度
        height, width = img.shape[:2]

        assert width == 256 and height == 85

        # # 计算裁剪区域的起始位置
        # start_h = int((height - 85) / 2)
        # end_h = start_h + 85

        # # 裁剪图像
        # cropped_image = img[start_h:end_h, :]

        # # 保存
        # cv2.imwrite(img_path, cropped_image)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="diffste_3859_blue") 
    parser.add_argument('--ocr_name', type=str, default="plate_cn_diffste_202308") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/")  
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.ocr_name, args.date_name)

    print("crop data.")
    print("date_name: {}".format(args.date_name))
    print("ocr_name: {}".format(args.ocr_name))
    print("input_dir: {}".format(args.input_dir))

    args.input_csv_path = os.path.join(args.input_dir, '{}.csv'.format(args.date_name))

    crop_data(args)