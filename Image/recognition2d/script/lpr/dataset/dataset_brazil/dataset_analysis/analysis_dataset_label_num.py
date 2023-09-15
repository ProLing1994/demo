import argparse
import cv2
import os
import pandas as pd
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from Image.recognition2d.script.lpr.dataset.dataset_brazil.dataset_dict.dataset_brazil_dict_normal import *


def analysis_dataset_label(args):

    # mkdir
    create_folder(args.output_analysis_dir)

    # init
    analysis_dict = {}

    # pd
    data_pd = pd.read_csv(args.input_csv_path)

    for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        # info
        img_path = row['img_path']
        json_path = row['json_path']
        plate_id = row['id'] 
        plate_name = row['name']
        plate_roi = row['roi'] 
        plate_color = row['color']
        plate_column = row['column']
        plate_num = row['num']

        # analysis info
        analysis_info = {}
        analysis_info["color"] = plate_color
        analysis_info["column"] = plate_column
        analysis_info["num"] = plate_num

        for key_name in analysis_label_columns.keys():
            
            for analysis_key in analysis_info.keys():

                if key_name != analysis_key:
                    continue
                
                if key_name == "num":
                    num = analysis_info[analysis_key]

                    for num_idx in range(len(num)):

                        if num[num_idx] not in analysis_label_columns[key_name]:
                            raise Exception

                        if key_name not in analysis_dict:
                            analysis_dict[key_name] = {}
                        
                        if num[num_idx] not in analysis_dict[key_name]:
                            analysis_dict[key_name][num[num_idx]] = 1
                        else:
                            analysis_dict[key_name][num[num_idx]] += 1

                else:

                    if analysis_info[analysis_key] not in analysis_label_columns[key_name]:
                        raise Exception

                    if key_name not in analysis_dict:
                        analysis_dict[key_name] = {}
                    
                    if analysis_info[analysis_key] not in analysis_dict[key_name]:
                        analysis_dict[key_name][analysis_info[analysis_key]] = 1
                    else:
                        analysis_dict[key_name][analysis_info[analysis_key]] += 1

    # write
    for key_name in analysis_label_columns.keys():
    
        if key_name not in analysis_dict:
            continue

        out_csv_path = os.path.join(args.output_analysis_dir, 'analysis_{}.csv'.format(key_name))
        analysis_pd = pd.DataFrame(analysis_dict[key_name], index=[args.date_name], columns=analysis_label_columns[key_name])
        analysis_pd.to_csv(out_csv_path, encoding="utf_8_sig")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="brazil_02210_202301") 
    parser.add_argument('--input_csv_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_csv/") 
    parser.add_argument('--output_analysis_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_analysis/") 
    args = parser.parse_args()

    args.input_csv_path = os.path.join(args.input_csv_dir, args.date_name + '.csv')
    args.output_analysis_dir = os.path.join(args.output_analysis_dir, args.date_name)

    print("analysis dataset label num.")
    print("date_name: {}".format(args.date_name))
    print("input_csv_path: {}".format(args.input_csv_path))

    analysis_dataset_label(args)