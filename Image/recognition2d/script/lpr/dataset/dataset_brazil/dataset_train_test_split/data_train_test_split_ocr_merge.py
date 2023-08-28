import argparse
import numpy as np
import os
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
# sys.path.insert(0, '/yuanhuan/demo')
from Image.Basic.utils.folder_tools import *

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d')
from script.lpr.dataset.dataset_cn.dataset_train_test_split.data_train_test_split_seg_merge import split


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ocr_name', type=str, default="plate_brazil_202309") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/")   
    args = parser.parse_args()
    
    args.input_dir = os.path.join(args.input_dir, args.ocr_name)

    print("data train test split merge.")
    print("input_dir: {}".format(args.input_dir))

    args.ignore_list = [
                        'ImageSetsOcrLabelNoAug', 
                        'ImageSetsOcrLabelNoAug_double_1st_line', 
                        'ImageSetsOcrLabelNoAug_double_2nd_line', 
                        'ImageSetsOcrLabelNoAug_single_line', 
                        'ImageSetsOcrLabel', 
                        'ImageSetsOcrLabel_double_1st_line', 
                        'ImageSetsOcrLabel_double_2nd_line', 
                        'ImageSetsOcrLabel_single_line', 
                        'ImageSetsBrazilnewstyleNoAug', 
                        'ImageSetsBrazilnewstyleNoAug_double_2nd_line', 
                        'ImageSetsBrazilnewstyleNoAug_double_1st_line', 
                        'ImageSetsBrazilnewstyleNoAug_single_line', 
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
    # dataset_cn_dict
    ###############################################
    ## no aug
    ###############################################
    args.label_name = ""
    args.bool_aug = False
    args.to_input_dir = os.path.join(args.input_dir, "ImageSetsOcrLabelNoAug")
    # args.to_input_dir = os.path.join(args.input_dir, "ImageSetsBrazilnewstyleNoAug")

    split(args)

    ###############################################
    ## with aug 
    ###############################################
    args.label_name = ""
    args.bool_aug = True
    args.to_input_dir = os.path.join(args.input_dir, "ImageSetsOcrLabel")

    split(args)
