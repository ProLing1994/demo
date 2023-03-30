import argparse
import os
import pandas as pd
import sys
import shutil
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

def find_data():
    data_list = get_sub_filepaths_suffix(args.input_dir, args.suffix)

    for data_idx in tqdm(range(len(data_list))):
        # init 
        input_path = data_list[data_idx]
        input_name = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, input_name)

        if (args.suffix in os.path.basename(input_path)):
            print(input_path, '->', output_path)
            shutil.copy(input_path, output_path)
            # shutil.move(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil/Brazil_motor/"
    # args.output_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil/Brazil_motor_refine/groundTruth/JPEGImages/"
    # args.suffix = ".jpg"
    args.output_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil/Brazil_motor_refine/groundTruth/Annotations_motor_Json/"
    args.suffix = ".json"

    find_data()