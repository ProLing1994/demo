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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/temp/UAE/"
    args.output_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_old_style/"

    args.input_data_list = ["check_0810_0811", "check_0828_0831", "check_0901_0903", "check_0904_0905", "check_1024_1029", "check_1024_1029_1080p", "check_1115_1116", "check_1115_1116_1080p"]

    args.img_folder = "JPEGImages"
    args.json_folder = "Json"


    for idx in tqdm(range(len(args.input_data_list))):

        folder_name = args.input_data_list[idx]
        input_dir = os.path.join(args.input_dir, folder_name)

        file_list = os.listdir(input_dir)
        for file_idx in tqdm(range(len(file_list))):
            file = file_list[file_idx]

            if file.endswith('.jpg'):
                img_path = os.path.join(input_dir, file)
                out_img_path = os.path.join(args.output_dir, folder_name.replace("check_", "uae_2022"), args.img_folder, file)
                # print(img_path, out_img_path)
                create_folder(os.path.dirname(out_img_path))
                shutil.copy(img_path, out_img_path)

            if file.endswith('.json'):
                img_path = os.path.join(input_dir, file)
                out_img_path = os.path.join(args.output_dir, folder_name.replace("check_", "uae_2022"), args.json_folder, file)
                # print(img_path, out_img_path)
                create_folder(os.path.dirname(out_img_path))
                shutil.copy(img_path, out_img_path)