import argparse
import cv2
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def data_reference(args):
    
    # mkdir
    create_folder(args.output_reference_dir)

    sub_folder_list = np.array(os.listdir(args.input_reference_dir))
    sub_folder_list = sub_folder_list[[not jpg.endswith('.jpg') for jpg in sub_folder_list]]

    for idx in tqdm(range(len(sub_folder_list))):
    
        sub_folder = os.path.join(args.input_reference_dir, sub_folder_list[idx])

        jpg_list = np.array(os.listdir(sub_folder))
        jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
        jpg_list.sort()
        
        for idy in tqdm(range(len(jpg_list))):

            input_reference_path = os.path.join(args.input_reference_dir, sub_folder_list[idx], jpg_list[idy])

            # mkdir
            output_reference_path = os.path.join(args.output_reference_dir, sub_folder_list[idx], jpg_list[idy])
            create_folder(os.path.dirname(output_reference_path))
            shutil.copy(input_reference_path, output_reference_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Pickup_middle_20230721") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/Taxi_remnant/training/sd_crop_clip_sam_bottle_0821/shenzhen") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/Taxi_remnant/training/sd_crop_clip_sam_bottle_0821/shenzhen") 
    args = parser.parse_args()

    args.img_suffix = ".jpg"

    print("data reference.")
    print("date_name: {}".format(args.date_name))

    args.input_dir = os.path.join(args.input_dir, args.date_name)
    args.input_reference_dir = os.path.join(args.input_dir, 'references')
    args.output_reference_dir = os.path.join(args.output_dir, 'references')

    data_reference(args)