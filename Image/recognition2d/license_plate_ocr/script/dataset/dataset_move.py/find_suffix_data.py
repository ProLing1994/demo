import argparse
import os
import pandas as pd
import sys
import shutil
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *


def find_data():
    data_list = get_sub_filepaths_suffix(args.input_dir, suffix='.jpg')

    for data_idx in range(len(data_list)):
        # init 
        input_path = data_list[data_idx]
        
        if args.ignore_suffix in input_path:
            continue

        license_plate_name = os.path.basename(os.path.dirname(input_path))
        licnese_plate_color = os.path.basename(os.path.dirname(os.path.dirname(input_path)))

        output_path = os.path.join(args.output_dir, licnese_plate_color, '{}_{}.jpg'.format(license_plate_name, data_idx))
        create_folder(os.path.dirname(output_path))

        print(input_path, '->', output_path)
        shutil.copy(input_path, output_path)
        # shutil.move(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_tq_5M_220418/5M_16mm/crop/"
    args.output_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_tq_5M_220418/5M_16mm/"
    args.ignore_suffix = '糊'

    find_data()