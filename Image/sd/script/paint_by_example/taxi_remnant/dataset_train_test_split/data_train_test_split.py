import argparse
import numpy as np
import os
import sys

from sklearn.model_selection import train_test_split

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

def split(args):
    
    # mkdir
    create_folder(os.path.dirname(args.trainval_file))

    img_list = np.array(get_sub_filepaths_suffix(args.img_dir, args.img_suffix))
    img_list = img_list[[os.path.basename(img).endswith(args.img_suffix) for img in img_list]]
    img_list = img_list[[os.path.exists(os.path.join(args.mask_dir, os.path.basename(img).replace(args.img_suffix, args.mask_suffix))) for img in img_list]]

    trainval_list, test_list = train_test_split(img_list, test_size=args.test_size, random_state=0)
    train_list, val_list = train_test_split(trainval_list, test_size=args.val_size, random_state=0)

    print("length: trainval: {}, train: {}, val: {}, test: {}, all: {}".format(len(trainval_list), len(train_list), len(val_list), len(test_list), (len(train_list) + len(val_list) + len(test_list))))
    with open(args.trainval_file, "w") as f:
        for img_path in trainval_list:
            img_name = os.path.basename(img_path)
            mask_name = img_name.replace(args.img_suffix, args.mask_suffix)
            mask_path = os.path.join(args.mask_dir, mask_name)
            ref_img_name = img_name.replace(args.img_suffix, args.ref_img_suffix)
            ref_img_path = os.path.join(args.ref_img_dir, ref_img_name)
            f.write('{} {} {}'.format(img_path, mask_path, ref_img_path))
            f.write("\n")

    with open(args.test_file, "w") as f:
        for img_path in test_list:
            img_name = os.path.basename(img_path)
            mask_name = img_name.replace(args.img_suffix, args.mask_suffix)
            mask_path = os.path.join(args.mask_dir, mask_name)
            ref_img_name = img_name.replace(args.img_suffix, args.ref_img_suffix)
            ref_img_path = os.path.join(args.ref_img_dir, ref_img_name)
            f.write('{} {} {}'.format(img_path, mask_path, ref_img_path))
            f.write("\n")

    with open(args.train_file, "w") as f:
        for img_path in train_list:
            img_name = os.path.basename(img_path)
            mask_name = img_name.replace(args.img_suffix, args.mask_suffix)
            mask_path = os.path.join(args.mask_dir, mask_name)
            ref_img_name = img_name.replace(args.img_suffix, args.ref_img_suffix)
            ref_img_path = os.path.join(args.ref_img_dir, ref_img_name)
            f.write('{} {} {}'.format(img_path, mask_path, ref_img_path))
            f.write("\n")

    with open(args.val_file, "w") as f:
        for img_path in val_list:
            img_name = os.path.basename(img_path)
            mask_name = img_name.replace(args.img_suffix, args.mask_suffix)
            mask_path = os.path.join(args.mask_dir, mask_name)
            ref_img_name = img_name.replace(args.img_suffix, args.ref_img_suffix)
            ref_img_path = os.path.join(args.ref_img_dir, ref_img_name)
            f.write('{} {} {}'.format(img_path, mask_path, ref_img_path))
            f.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Pickup_middle_20230721") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/Taxi_remnant/training/sd_crop_clip_sam_bottle_0821/shenzhen") 
    parser.add_argument('--img_folder', type=str, default="JPEGImages_resize") 
    parser.add_argument('--mask_folder', type=str, default="masks_sam_resize") 
    parser.add_argument('--ref_img_folder', type=str, default="references_resize") 
    args = parser.parse_args()

    args.img_suffix = ".jpg"
    args.mask_suffix = ".jpg"
    args.ref_img_suffix = ".jpg"
    args.img_dir = os.path.join(args.input_dir, args.date_name, args.img_folder)
    args.mask_dir = os.path.join(args.input_dir, args.date_name, args.mask_folder)
    args.ref_img_dir = os.path.join(args.input_dir, args.date_name, args.ref_img_folder)

    print("data train test split.")
    print("date_name: {}".format(args.date_name))
    print("img_dir: {}".format(args.img_dir))
    print("mask_dir: {}".format(args.mask_dir))
    print("ref_img_dir: {}".format(args.ref_img_dir))

    args.trainval_file = os.path.join(args.input_dir, args.date_name, "ImageSets/Main/trainval.txt")
    args.train_file = os.path.join(args.input_dir, args.date_name, "ImageSets/Main/train.txt")
    args.val_file = os.path.join(args.input_dir, args.date_name, "ImageSets/Main/val.txt")
    args.test_file = os.path.join(args.input_dir, args.date_name, "ImageSets/Main/test.txt")

    args.test_size = 0.05
    args.val_size = 0.05

    split(args)