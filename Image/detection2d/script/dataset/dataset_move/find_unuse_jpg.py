import argparse
import os
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *


def find_unuse_jpg(args):
    # mkdir
    create_folder(args.output_dir)
    
    input_list = get_sub_filepaths_suffix(args.input_dir, suffix='.jpg')
    input_list = [os.path.basename(input_list[idx]) for idx in range(len(input_list))]
    print(len(input_list))

    find_input_list = get_sub_filepaths_suffix(args.find_input_dir, suffix='.jpg')
    find_input_list = [os.path.basename(find_input_list[idx]) for idx in range(len(find_input_list))]
    print(len(find_input_list))
    find_input_list = [find_input_list[idx] for idx in range(len(find_input_list)) if find_input_list[idx] not in input_list]
    print(len(find_input_list))
    # print(find_input_list)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/mnt/huanyuan2/data/image/3D_huoche/3D_huoche/"
    args.find_input_dir = "/mnt/huanyuan2/data/image/weinan/20210615/"
    args.output_dir = "/mnt/huanyuan2/data/image/weinan/not_in_3D_huoche/"

    find_unuse_jpg(args)