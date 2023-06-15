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
            input_merge_file_idy = os.path.join(merge_dir_idx, args.label_name, merge_file_idy)

            with open(input_merge_file_idy, "r") as f:
                for line in f:
                    merge_list[idy].append(line.strip())
            
            if args.bool_aug:
                input_merge_file_idy = os.path.join(merge_dir_idx, args.label_name, 'ImageSets_aug', merge_file_idy)

                with open(input_merge_file_idy, "r") as f:
                    for line in f:
                        merge_list[idy].append(line.strip())
    
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
    parser.add_argument('--seg_name', type=str, default="seg_zd_202306") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/")   
    args = parser.parse_args()
    
    args.input_dir = os.path.join(args.input_dir, args.seg_name)

    print("data train test split merge.")
    print("input_dir: {}".format(args.input_dir))

    args.ignore_list = [
                        'ImageSetsColorLabelNoAug',
                        'ImageSetsColorLabel',
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
    # dataset_zd_dict_city
    ###############################################
    ## no aug
    ###############################################
    args.label_name = "city_label"
    args.bool_aug = False
    args.to_input_dir = os.path.join(args.input_dir, "ImageSetsColorLabelNoAug", args.label_name)

    split(args)

    ###############################################
    ## with aug 
    ###############################################
    args.label_name = "city_label"
    args.bool_aug = True
    args.to_input_dir = os.path.join(args.input_dir, "ImageSetsColorLabel", args.label_name)

    split(args)

    ###############################################
    # dataset_zd_dict_color
    ###############################################
    ## no aug
    ###############################################
    args.label_name = "color_label"
    args.bool_aug = False
    args.to_input_dir = os.path.join(args.input_dir, "ImageSetsColorLabelNoAug", args.label_name)

    split(args)

    ###############################################
    ## with aug 
    ###############################################
    args.label_name = "color_label"
    args.bool_aug = True
    args.to_input_dir = os.path.join(args.input_dir, "ImageSetsColorLabel", args.label_name)

    split(args)


    ###############################################
    # dataset_zd_dict_city & dataset_zd_dict_color
    ###############################################
    ## no aug
    ###############################################
    args.label_name = "city_color_label"
    args.bool_aug = False
    args.to_input_dir = os.path.join(args.input_dir, "ImageSetsColorLabelNoAug", args.label_name)

    split(args)

    ###############################################
    ## with aug 
    ###############################################
    args.label_name = "city_color_label"
    args.bool_aug = True
    args.to_input_dir = os.path.join(args.input_dir, "ImageSetsColorLabel", args.label_name)

    split(args)