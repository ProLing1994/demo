import argparse
import cv2
import json
import numpy as np
import os
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def gen_seg_mask(args):

    # mkdir
    create_folder(args.mask_dir)
    create_folder(args.mask_img_dir)

    # img_list
    img_list = get_sub_filepaths_suffix(args.img_dir, args.img_suffix)
    img_list.sort()

    for idx in tqdm(range(len(img_list))):

        img_path = img_list[idx]
        img_name = os.path.basename(img_path)

        mask_name = img_name.replace(args.img_suffix, args.mask_suffix)
        output_mask_path = os.path.join(args.mask_dir, mask_name)
        output_mask_img_path = os.path.join(args.mask_img_dir, mask_name)

        # img
        img = cv2.imread(img_path)

        # mask_img
        mask = np.zeros(img.shape, dtype=img.dtype)
        mask_img = np.zeros(img.shape, dtype=img.dtype)
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.5, gamma=0.)
        
        cv2.imwrite(output_mask_path, mask)
        cv2.imwrite(output_mask_img_path, mask_img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="beijing_disable") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/HY_Tanker/original") 
    parser.add_argument('--img_folder', type=str, default="JPEGImages") 
    parser.add_argument('--mask_folder', type=str, default="mask") 
    parser.add_argument('--mask_img_folder', type=str, default="mask_img") 
    args = parser.parse_args()

    args.img_suffix = ".jpg"
    args.mask_suffix = ".png"
    args.img_dir = os.path.join(args.input_dir, args.date_name, args.img_folder)
    args.mask_dir = os.path.join(args.input_dir, args.date_name, args.mask_folder)
    args.mask_img_dir = os.path.join(args.input_dir, args.date_name, args.mask_img_folder)

    print("gen seg mask.")
    print("date_name: {}".format(args.date_name))
    print("img_dir: {}".format(args.img_dir))
    print("mask_dir: {}".format(args.mask_dir))
    print("mask_img_dir: {}".format(args.mask_img_dir))

    gen_seg_mask(args)