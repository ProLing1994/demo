import argparse
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
# sys.path.insert(0, '/yuanhuan/demo')
from Image.Basic.utils.folder_tools import *

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from Image.recognition2d.script.lpr.dataset.dataset_brazil.dataset_dict.dataset_brazil_dict_normal import *


def merge_analysis_pd(args):

    # mkdir
    create_folder(args.output_analysis_dir)

    # init
    analysis_dict = {}

    # merge
    for idx in tqdm(range(len(args.analysis_dir_list))):
        analysis_dir = args.analysis_dir_list[idx]

        for key_name in analysis_label_columns.keys():

            analysis_name = 'analysis_{}.csv'.format(key_name)
            analysis_path = os.path.join(analysis_dir, analysis_name)
            analysis_pd = pd.read_csv(analysis_path, encoding='utf_8_sig')

            if key_name not in analysis_dict:
                analysis_dict[key_name] = analysis_pd
            else:
                analysis_dict[key_name] = pd.concat([analysis_dict[key_name], analysis_pd]) 

    # write
    for key_name in analysis_label_columns.keys():

        if key_name not in analysis_dict:
            continue

        out_csv_path = os.path.join(args.output_analysis_dir, 'analysis_{}.csv'.format(key_name))
        analysis_pd = pd.DataFrame(analysis_dict[key_name])
        analysis_pd.to_csv(out_csv_path, index=False, encoding="utf_8_sig")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_analysis/")   
    args = parser.parse_args()

    print("analysis dataset label num merge.")
    print("input_dir: {}".format(args.input_dir))

    args.ignore_list = [
                        'ImageSetsLabel',
                        ]

    dir_list = np.array(os.listdir(args.input_dir))
    dir_list = list(dir_list[[dir not in args.ignore_list for dir in dir_list]])
    dir_list = [os.path.join(args.input_dir, dir) for dir in dir_list]   
    args.analysis_dir_list = dir_list

    args.output_analysis_dir = os.path.join(args.input_dir, "ImageSetsLabel")
    merge_analysis_pd(args)