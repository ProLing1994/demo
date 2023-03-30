import argparse
import cv2
import numpy as np
import os
import sys 
import shutil
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *


def find_size_img(args):
    # mkdir 
    create_folder(args.output_jpg_dir)

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list.sort()

    for idx in tqdm(range(len(jpg_list))):
        # jpg
        jpg_name = jpg_list[idx]
        jpg_path = os.path.join(args.jpg_dir, jpg_name)
    
        # output jpg
        output_jpg_path = os.path.join(args.output_jpg_dir, jpg_name)

        img = cv2.imread(jpg_path)

        img_width = img.shape[1]
        img_height = img.shape[0]

        if img_width != args.jpg_width or img_height != args.jpg_heght:
            print("{} -> {}".format(jpg_path, output_jpg_path))
            shutil.move(jpg_path, output_jpg_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_C28_detection/america_new/temp/20230308/") 
    args = parser.parse_args()

    args.jpg_dir = os.path.join(args.input_dir, "JPEGImages/")
    args.output_jpg_dir = os.path.join(args.input_dir, "JPEGImages_unknow/")

    args.jpg_width = 1920
    args.jpg_heght = 1080

    find_size_img(args)