import argparse
import importlib
import os
import pandas as pd
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')


def analysis_dataset_label(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module(args.seg_dict_name) 

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
        plate_country = row['country'] 
        plate_city = row['city'] 
        plate_color = row['color'] 
        plate_column = row['column'] 
        plate_num = row['num'] 
        crop_img_path = row['crop_img'] 
        crop_xml_path = row['crop_xml'] 
        if 'crop_json' not in row:
            crop_json_path = ""
        else:
            crop_json_path = row['crop_json'] 

        # load object_roi_list
        object_roi_list = dataset_dict.load_ori_object_roi(crop_xml_path, crop_json_path, args.new_style)

        for key_name in dataset_dict.analysis_crop_label_columns.keys():
    
            for idz in range(len(object_roi_list)):
                classname = object_roi_list[idz]["classname"]
                bndbox = object_roi_list[idz]["bndbox"]

                if classname not in dataset_dict.analysis_crop_label_columns[key_name]:
                    continue

                if key_name not in analysis_dict:
                    analysis_dict[key_name] = {}
                
                if classname not in analysis_dict[key_name]:
                    analysis_dict[key_name][classname] = 1
                else:
                    analysis_dict[key_name][classname] += 1
    
    for key_name in dataset_dict.analysis_crop_label_columns.keys():

        if key_name not in analysis_dict:
            continue

        out_csv_path = os.path.join(args.output_analysis_dir, 'analysis_{}.csv'.format(key_name))
        analysis_pd = pd.DataFrame(analysis_dict[key_name], index=[args.date_name], columns=dataset_dict.analysis_crop_label_columns[key_name])
        analysis_pd.to_csv(out_csv_path, encoding="utf_8_sig")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="shate_20230308") 
    parser.add_argument('--input_csv_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_crop_csv/") 
    parser.add_argument('--output_analysis_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_analysis_crop/") 
    parser.add_argument('--new_style', action='store_true', default=False) 
    args = parser.parse_args()

    args.seg_dict_name = "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_normal"
    args.input_csv_path = os.path.join(args.input_csv_dir, args.date_name + '.csv')
    args.output_analysis_dir = os.path.join(args.output_analysis_dir, args.date_name)

    print("analysis dataset label num.")
    print("date_name: {}".format(args.date_name))
    print("input_csv_path: {}".format(args.input_csv_path))

    analysis_dataset_label(args)