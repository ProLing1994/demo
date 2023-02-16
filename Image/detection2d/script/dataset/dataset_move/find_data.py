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

        if (args.suffix in os.path.basename(input_path)):
            print(input_path, '->', output_path)
            # shutil.copy(input_path, output_path)
            shutil.move(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/RM_BSD/wideangle_2022_2023/JPEGImages/"
    args.output_dir = "/yuanhuan/data/image/RM_BSD/wideangle_2022_2023/JSON/"
    args.suffix = ".json"

    find_data()