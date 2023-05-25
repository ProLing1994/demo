import argparse
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

def prepare_dataset(args):

    # mkdir
    create_folder(args.output_img_dir)
    create_folder(args.output_json_dir)
    create_folder(os.path.dirname(args.output_txt_file))

    # img list 
    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()
    print(len(img_list))

    # file list 
    file_list = []

    for idx in tqdm(range(len(img_list))):

        img_path = img_list[idx]
        json_path = img_path.replace(".jpg", ".json").replace(args.input_img_dir, args.input_json_dir)

        img_name = os.path.basename(img_path)   
        json_name = img_name.replace(".jpg", ".json")

        out_img_path = os.path.join(args.output_img_dir, img_name)
        output_json_dir = os.path.join(args.output_json_dir, json_name)

        if os.path.exists(img_path) and os.path.exists(json_path):
            shutil.copy(img_path, out_img_path)
            shutil.copy(json_path, output_json_dir)
            file_list.append(img_name)
        else:
            print(img_path)
            print(json_path)
    
    with open(args.output_txt_file, "w") as f:
        for jpg in file_list:
            f.write(jpg.replace(".jpg", ""))
            f.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dit = "/yuanhuan/data/image/Open_Source/Wider_Face/original/WIDER_val/"
    args.input_img_dir = os.path.join(args.input_dit, "images")
    args.input_json_dir = os.path.join(args.input_dit, "json_landmark")

    args.output_dir = "/yuanhuan/data/image/Open_Source/Wider_Face/training_landmark/"
    args.output_img_dir = os.path.join(args.output_dir, "JPEGImages")
    args.output_json_dir = os.path.join(args.output_dir, "Annotations_json")

    args.output_txt_file = args.output_dir + "ImageSets/Main/val.txt"

    prepare_dataset(args)

    args.input_dit = "/yuanhuan/data/image/Open_Source/Wider_Face/original/WIDER_train/"
    args.input_img_dir = os.path.join(args.input_dit, "images")
    args.input_json_dir = os.path.join(args.input_dit, "json_landmark")

    args.output_dir = "/yuanhuan/data/image/Open_Source/Wider_Face/training_landmark/"
    args.output_img_dir = os.path.join(args.output_dir, "JPEGImages")
    args.output_json_dir = os.path.join(args.output_dir, "Annotations_json")

    args.output_txt_file = args.output_dir + "ImageSets/Main/train.txt"

    prepare_dataset(args)