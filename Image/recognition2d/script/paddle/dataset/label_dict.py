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

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')

def gen_label_dict(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module( args.data_dict_name ) 
    ocr_labels = dataset_dict.kind_num_labels
    # run
    output_label_txt = os.path.join(args.output_dir, args.output_name)
    with open(output_label_txt, "w") as f:
        for idx in range(len(ocr_labels)):
            f.write('{}'.format(ocr_labels[idx]))
            f.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/model/image/lpr/paddle_dict/plate_cn_202305") 
    parser.add_argument('--output_name', type=str, default="cn_dict.txt") 
    parser.add_argument('--data_dict_name', type=str, default="script.lpr.dataset.dataset_cn.dataset_dict.dataset_cn_dict_normal") 

    args = parser.parse_args()

    gen_label_dict(args)
