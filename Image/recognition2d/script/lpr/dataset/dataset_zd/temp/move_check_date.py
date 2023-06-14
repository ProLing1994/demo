import argparse
import cv2
import io
import json
import os
import pandas as pd
import shutil
import sys
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def find_img_path(img_name, img_list):
    
    for idx in range(len(img_list)):
        img_path = img_list[idx]

        if os.path.basename(img_path) == img_name:
            return img_path
        
    return None


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.ori_input_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/temp/UAE/"
    args.input_crop_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/temp/UAE_crop/"
    args.output_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_old_style/"

    # args.input_data_list = ["check_crop_0", "check_crop_1", "check_crop_2", "check_crop_3", "check_crop_4", "check_crop_5", "check_crop_6", "check_crop_7", "check_crop_8", \
    #                             "check_crop_city_AJMAN_0", "check_crop_city_FUJAIRAH_0", "check_crop_city_RAK_0", "check_crop_city_SHARJAH_0", "check_crop_city_UMMALQAIWAIN_0", "check_crop_color_green_0", "check_crop_color_yellow_0"]
    args.input_data_list = [ "check_crop_city_AJMAN_0", "check_crop_city_FUJAIRAH_0", "check_crop_city_RAK_0", "check_crop_city_SHARJAH_0", "check_crop_city_UMMALQAIWAIN_0", "check_crop_color_green_0", "check_crop_color_yellow_0"]

    args.input_img_folder = "Images"
    args.output_img_folder = "JPEGImages"
    args.output_json_folder = "Json"

    # ori img list 
    ori_img_list = get_sub_filepaths_suffix(args.ori_input_dir, ".jpg")
    ori_img_list.sort()
    
    for idx in tqdm(range(len(args.input_data_list))):

        input_data_name = args.input_data_list[idx]
        input_dir = os.path.join(args.input_crop_dir, input_data_name, args.input_img_folder)

        img_list = os.listdir(input_dir)
        for img_idx in tqdm(range(len(img_list))):
            img = img_list[img_idx]
            img_name = ("-").join(img.split("-")[:-1]) + '.jpg'
            img_path = os.path.join(input_dir, img)

            ori_img_path = find_img_path(img_name, ori_img_list)
            ori_json_path = ori_img_path.replace(".jpg", ".json")

            out_img_path = os.path.join(args.output_dir, input_data_name.replace("check_crop_", "uae_2022_"), args.output_img_folder, os.path.basename(ori_img_path))
            # print(img_path, out_img_path)
            create_folder(os.path.dirname(out_img_path))
            shutil.copy(ori_img_path, out_img_path)

            out_json_path = os.path.join(args.output_dir, input_data_name.replace("check_crop_", "uae_2022_"), args.output_json_folder, os.path.basename(ori_json_path))
            # print(ori_json_path, out_json_path)
            create_folder(os.path.dirname(out_json_path))
            try:
                shutil.copy(ori_json_path, out_json_path)
            except:
                print(ori_json_path)