import argparse
import os
import shutil
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
# sys.path.insert(0, '/yuanhuan/demo')
from Image.Basic.utils.folder_tools import *


def lpr_to_paddleocr_label(args):

    # input list 
    input_list = []
    input_path = os.path.join(args.input_dir, args.txt_dir, "{}.txt".format(args.format))
    with open(input_path, "r") as f:
        for line in f:
            input_list.append(line.strip())
    
    # output path
    if args.format == "train":
        output_folder = os.path.join(args.output_dir, "train_data", "rec")
        output_label_txt = os.path.join(output_folder, "rec_gt_train.txt")
        output_data_folder = os.path.join(output_folder, "train")
        output_data_relative_folder = os.path.join("train_data", "rec", "train")
    elif args.format == "test":
        output_folder = os.path.join(args.output_dir, "train_data", "rec")
        output_label_txt = os.path.join(output_folder, "rec_gt_test.txt")
        output_data_folder = os.path.join(output_folder, "test")
        output_data_relative_folder = os.path.join("train_data", "rec", "test")

    # mkdir
    create_folder(output_data_folder)

    # run
    with open(output_label_txt, "w") as f:
        for idx in tqdm(range(len(input_list))):

            input_path = input_list[idx]
            input_name = os.path.basename(input_path)
            input_label = str(input_name).split('.jpg')[0].split('_')[-1]

            output_path = os.path.join(output_data_folder, input_name)
            output_relative_path = os.path.join(output_data_relative_folder, input_name)

            f.write('{}'.format(output_relative_path))
            f.write("\t")
            f.write('{}'.format(input_label))
            f.write("\n")

            shutil.copy(input_path, output_path)
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/ocr_merge_test/"
    args.txt_dir = "ImageSets/Main/"

    args.output_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_paddle_ocr"
    args.format = "train"

    lpr_to_paddleocr_label(args)
    