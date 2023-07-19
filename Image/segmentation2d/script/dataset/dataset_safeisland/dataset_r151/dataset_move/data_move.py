import argparse
import os
import sys

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.segmentation2d.script.dataset.dataset_safeisland.dataset_c28.dataset_move.data_move import data_move


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="avm_right_20230402_segonly") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_R151_safeisland/original") 
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