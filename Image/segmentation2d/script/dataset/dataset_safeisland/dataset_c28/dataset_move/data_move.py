import argparse
import os
import shutil
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def data_move(args):

    # mkdir
    create_folder(args.output_trainval_dir)
    create_folder(args.output_test_dir)

    with open(args.trainval_file, "r") as f:
        lines = f.readlines()
        for idx in tqdm(range(len(lines))):
            file = lines[idx].strip()
            img_path = file.split(".jpg ")[0] + args.img_suffix
            mask_path = file.split(".jpg ")[1]

            img_name = os.path.basename(img_path)
            mask_name = os.path.basename(mask_path)
            output_img_path = os.path.join(args.output_trainval_dir, img_name)
            output_mask_path = os.path.join(args.output_trainval_dir, mask_name)

            shutil.copy(img_path, output_img_path)
            shutil.copy(mask_path, output_mask_path)

    with open(args.test_file, "r") as f:
        lines = f.readlines()
        for idx in tqdm(range(len(lines))):
            file = lines[idx].strip()
            img_path = file.split(".jpg ")[0] + args.img_suffix
            mask_path = file.split(".jpg ")[1]

            img_name = os.path.basename(img_path)
            mask_name = os.path.basename(mask_path)
            output_img_path = os.path.join(args.output_test_dir, img_name)
            output_mask_path = os.path.join(args.output_test_dir, mask_name)

            shutil.copy(img_path, output_img_path)
            shutil.copy(mask_path, output_mask_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="america") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_C28_safeisland/original/") 
    parser.add_argument('--seg_name', type=str, default="safeisland_mask_202307")
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_C28_safeisland/training/") 
    args = parser.parse_args()

    args.img_suffix = ".jpg"
    args.mask_suffix = ".png"

    print("data move.")
    print("date_name: {}".format(args.date_name))

    args.trainval_file = os.path.join(args.input_dir, args.date_name, "ImageSets/Main/trainval.txt")
    args.train_file = os.path.join(args.input_dir, args.date_name, "ImageSets/Main/train.txt")
    args.val_file = os.path.join(args.input_dir, args.date_name, "ImageSets/Main/val.txt")
    args.test_file = os.path.join(args.input_dir, args.date_name, "ImageSets/Main/test.txt")
    
    args.output_trainval_dir = os.path.join(args.output_dir, args.seg_name, "trainval")
    args.output_test_dir = os.path.join(args.output_dir, args.seg_name, "test")

    data_move(args)