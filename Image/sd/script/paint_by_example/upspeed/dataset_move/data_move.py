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
    create_folder(args.output_img_train_dir)
    create_folder(args.output_img_val_dir)
    create_folder(args.output_img_test_dir)
    create_folder(args.output_bbox_train_dir)
    create_folder(args.output_bbox_val_dir)
    create_folder(args.output_bbox_test_dir)

    with open(args.train_file, "r") as f:
        lines = f.readlines()
        for idx in tqdm(range(len(lines))):
            file = lines[idx].strip()
            img_path = file.split(".jpg ")[0] + args.img_suffix
            bbox_path = file.split(".jpg ")[1]

            img_name = os.path.basename(img_path)
            bbox_name = os.path.basename(bbox_path)
            output_img_path = os.path.join(args.output_img_train_dir, img_name)
            output_bbox_path = os.path.join(args.output_bbox_train_dir, bbox_name)

            shutil.copy(img_path, output_img_path)
            shutil.copy(bbox_path, output_bbox_path)

    with open(args.val_file, "r") as f:
        lines = f.readlines()
        for idx in tqdm(range(len(lines))):
            file = lines[idx].strip()
            img_path = file.split(".jpg ")[0] + args.img_suffix
            bbox_path = file.split(".jpg ")[1]

            img_name = os.path.basename(img_path)
            bbox_name = os.path.basename(bbox_path)
            output_img_path = os.path.join(args.output_img_val_dir, img_name)
            output_bbox_path = os.path.join(args.output_bbox_val_dir, bbox_name)

            shutil.copy(img_path, output_img_path)
            shutil.copy(bbox_path, output_bbox_path)

    with open(args.test_file, "r") as f:
        lines = f.readlines()
        for idx in tqdm(range(len(lines))):
            file = lines[idx].strip()
            img_path = file.split(".jpg ")[0] + args.img_suffix
            bbox_path = file.split(".jpg ")[1]

            img_name = os.path.basename(img_path)
            bbox_name = os.path.basename(bbox_path)
            output_img_path = os.path.join(args.output_img_test_dir, img_name)
            output_bbox_path = os.path.join(args.output_bbox_test_dir, bbox_name)

            shutil.copy(img_path, output_img_path)
            shutil.copy(bbox_path, output_bbox_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Europe") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_upspeed/training/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_upspeed/training/") 
    args = parser.parse_args()

    args.img_suffix = ".jpg"
    args.bbox_suffix = ".jpg"

    print("data move.")
    print("date_name: {}".format(args.date_name))

    args.trainval_file = os.path.join(args.input_dir, args.date_name, "ImageSets/Main/trainval.txt")
    args.train_file = os.path.join(args.input_dir, args.date_name, "ImageSets/Main/train.txt")
    args.val_file = os.path.join(args.input_dir, args.date_name, "ImageSets/Main/val.txt")
    args.test_file = os.path.join(args.input_dir, args.date_name, "ImageSets/Main/test.txt")
    
    args.output_img_train_dir = os.path.join(args.output_dir, args.date_name, "data", "images", "train_0")
    args.output_img_val_dir = os.path.join(args.output_dir, args.date_name, "data", "images", "validation")
    args.output_img_test_dir = os.path.join(args.output_dir, args.date_name, "data", "images", "test")

    args.output_bbox_train_dir = os.path.join(args.output_dir, args.date_name, "data", "bbox", "train_0")
    args.output_bbox_val_dir = os.path.join(args.output_dir, args.date_name, "data", "bbox", "validation")
    args.output_bbox_test_dir = os.path.join(args.output_dir, args.date_name, "data", "bbox", "test")

    data_move(args)