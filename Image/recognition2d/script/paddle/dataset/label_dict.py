import argparse
import importlib
import os
import shutil
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
# sys.path.insert(0, '/yuanhuan/demo')
from Image.Basic.utils.folder_tools import *

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')

def gen_label_dict(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.dataset.dataset_seg_zd.dataset_dict.' + args.seg_dict_name) 
    ocr_labels = dataset_dict.kind_num_labels

    # run
    output_label_txt = os.path.join(args.output_dir, args.output_name)
    with open(output_label_txt, "w") as f:
        for idx in range(len(ocr_labels)):
            f.write('{}'.format(ocr_labels[idx]))
            f.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.seg_dict_name = "dataset_zd_dict_nomal"

    args.output_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_paddle_ocr"
    args.output_name = "zd_dict.txt"

    gen_label_dict(args)
