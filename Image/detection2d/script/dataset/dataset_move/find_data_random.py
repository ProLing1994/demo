import argparse
import os
import random
import sys
import shutil
from tqdm import tqdm


sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def find_data_random(args):

    # mkdir 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for idx in tqdm(range(len(args.input_dir_list))):

        input_dir = args.input_dir_list[idx]

        jpg_list = get_sub_filepaths_suffix(input_dir, suffix='.jpg')

        for idy in tqdm(range(len(jpg_list))):
            jpg_name = os.path.basename(jpg_list[idy])
            input_path = os.path.join(input_dir, jpg_name)
            output_path = os.path.join(args.output_dir, jpg_name)

            if random.random() > 0.98:

                print(input_path, '->', output_path)
                shutil.copy(input_path, output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir_list = [
                            "/yuanhuan/data/image/RM_C28_detection/america/JPEGImages_test/",
                            "/yuanhuan/data/image/RM_C28_detection/canada/JPEGImages_test/",
                            "/yuanhuan/data/image/RM_C28_detection/finished/JPEGImages_test/",
                            "/yuanhuan/data/image/RM_C28_detection/safezone/JPEGImages_test/",
                            "/yuanhuan/data/image/RM_C28_detection/zhongdong/JPEGImages_test/"
    ]

    args.output_dir = "/yuanhuan/data/image/RM_C28_detection/Quantitative_Image"

    find_data_random(args)