import argparse
import os
import shutil
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def dataset_check(args):

    # mkdir
    create_folder(args.img_dir)
    create_folder(args.mask_dir)

    # img list 
    img_list = get_sub_filepaths_suffix(args.date_dir, args.img_suffix)
    img_list.sort()
    print(len(img_list))

    for idx in tqdm(range(len(img_list))):

        img_path = img_list[idx]
        img_dir = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)
        
        mask_name = img_name.replace(args.img_suffix, args.mask_suffix)
        mask_path = os.path.join(img_dir, mask_name)

        output_img_path = os.path.join(args.img_dir, img_name)
        print(img_path, output_img_path)
        shutil.move(img_path, output_img_path)

        if os.path.exists(mask_path):
            output_mask_path = os.path.join(args.mask_dir, mask_name)
            print(mask_path, output_mask_path)
            shutil.move(mask_path, output_mask_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="avm_right_20230402_segonly") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_R151_safeisland/original") 
    parser.add_argument('--img_folder', type=str, default="JPEGImages") 
    parser.add_argument('--mask_folder', type=str, default="mask") 

    args = parser.parse_args()

    args.img_suffix = ".jpg"
    args.mask_suffix = ".png"
    args.date_dir = os.path.join(args.input_dir, args.date_name)
    args.img_dir = os.path.join(args.input_dir, args.date_name, args.img_folder)
    args.mask_dir = os.path.join(args.input_dir, args.date_name, args.mask_folder)

    print("dataset check.")
    print("date_name: {}".format(args.date_name))
    print("img_dir: {}".format(args.img_dir))
    print("mask_dir: {}".format(args.mask_dir))

    # dataset check
    dataset_check(args)