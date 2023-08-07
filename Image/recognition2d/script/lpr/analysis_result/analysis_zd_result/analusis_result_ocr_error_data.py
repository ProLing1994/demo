import argparse
import os
from re import I
import pandas as pd
import sys 
import shutil
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def find_result_jpg(args):
    # mkdir
    create_folder(args.output_error_data_dir)

    data_pd = pd.read_csv(args.input_error_data_csv)

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
        
        img_path = row['file']

        for key in args.find_dict.keys():
            key_res = row[key]

            if key_res == args.find_dict[key]:

                # img
                # to_img_name = os.path.basename(img_path)
                to_img_name = os.path.basename(img_path).replace('.jpg', '_{}.jpg'.format(row['ocr']))
                to_img_path = os.path.join(args.output_error_data_dir, key, to_img_name)
                create_folder(os.path.dirname(to_img_path))
                # shutil.move(img_path, to_img_path)
                shutil.copy(img_path, to_img_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # args.input_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_pad_20230703"
    
    # args.input_error_data_csv = os.path.join(args.input_dir, "test", "ocr_merge_test_result.csv")
    # args.output_error_data_dir = os.path.join(args.input_dir, "test", "error_data", "ocr_merge_test")
    
    # args.find_dict = {'res': 0}

    args.input_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230727_diffste_1_7M"
    
    args.input_error_data_csv = os.path.join(args.input_dir, "inference_iter_epoch_50", "test_caffe", "data_all_no_xianggangaomen_doubleyellow_test_result.csv")
    args.output_error_data_dir = os.path.join(args.input_dir, "inference_iter_epoch_50", "error_data", "data_all_no_xianggangaomen_doubleyellow_test_result")
    
    args.find_dict = {'res': 0}
    
    find_result_jpg(args)
