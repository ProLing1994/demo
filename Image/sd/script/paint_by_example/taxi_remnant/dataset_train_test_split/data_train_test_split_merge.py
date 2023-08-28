import argparse
import numpy as np
import os
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
# sys.path.insert(0, '/yuanhuan/demo')
from Image.Basic.utils.folder_tools import *


def split(args):
    
    # init
    merge_list = []
    for idy in range(len(args.merge_file_list)):
        merge_list.append([])

    # merge
    for idx in tqdm(range(len(args.merge_dir_list))):
        merge_dir_idx = args.merge_dir_list[idx]

        for idy in range(len(args.merge_file_list)):
            merge_file_idy = args.merge_file_list[idy]
            input_merge_file_idy = os.path.join(merge_dir_idx, merge_file_idy)
            
            try:
                with open(input_merge_file_idy, "r") as f:
                    for line in f:
                        merge_list[idy].append(line.strip())
            except:
                print(input_merge_file_idy)
                continue
    
    # output
    for idy in range(len(args.merge_file_list)):
        merge_file_idy = args.merge_file_list[idy]
        output_merge_file_idy = os.path.join(args.to_input_dir, merge_file_idy)

        # mkdir
        create_folder(os.path.dirname(output_merge_file_idy))

        with open(output_merge_file_idy, "w") as f:
            for jpg in merge_list[idy]:
                f.write('{}'.format(jpg))
                f.write("\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/Taxi_remnant/training/sd_crop_clip_sam_bottle_0821/shenzhen")   
    args = parser.parse_args()

    print("data train test split merge.")
    print("input_dir: {}".format(args.input_dir))

    args.ignore_list = [
                        'ImageSets',
                        'references',
                        ]

    args.merge_file_list = [
                        "ImageSets/Main/trainval.txt", 
                        "ImageSets/Main/train.txt",
                        "ImageSets/Main/val.txt",
                        "ImageSets/Main/test.txt",
                        ]

    dir_list = np.array(os.listdir(args.input_dir))
    dir_list = list(dir_list[[dir not in args.ignore_list for dir in dir_list]])
    dir_list = [os.path.join(args.input_dir, dir) for dir in dir_list]   
    args.merge_dir_list = dir_list


    ###############################################
    # dataset
    ###############################################
    args.to_input_dir = os.path.join(args.input_dir, "ImageSets")

    split(args)
