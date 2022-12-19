import argparse
import os
import pandas as pd
import sys
import shutil
from tqdm import tqdm


def find_data():
    data_list = os.listdir(args.input_dir)

    for data_idx in range(len(data_list)):
        # init 
        input_path = os.path.join(args.input_dir, data_list[data_idx])
        output_path = os.path.join(args.output_dir, data_list[data_idx])

        if args.file_format in input_path:
            print(input_path, '->', output_path)
            # shutil.copy(input_path, output_path)
            shutil.move(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/test/视频_20220310/264原始视频/"
    # args.output_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/test/视频_20220310/264原始视频/2M/"
    args.output_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/test/视频_20220310/264原始视频/5M/"

    # 5M: 000005000390, 2M: 000006000390
    args.file_format = "000005000390"
    find_data()