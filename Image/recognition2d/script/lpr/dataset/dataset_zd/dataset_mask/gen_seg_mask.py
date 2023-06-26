import argparse
import cv2
import importlib
import os
import pandas as pd
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from script.lpr.dataset.dataset_cn.dataset_mask.utils import draw_mask


def gen_seg_mask(args):

    # dataset_zd_dict
    dataset_dict = importlib.import_module('script.lpr.dataset.dataset_zd.dataset_dict.' + args.seg_dict_name) 

    # mkdir
    create_folder(args.output_img_dir)
    create_folder(args.output_mask_dir)
    create_folder(args.output_mask_img_dir)
    create_folder(args.output_bbox_img_dir)

    # init 
    csv_list = []           # [{"img_path": "", "json_path": "", "roi_img_path": "", "roi_mask_path": "", "id": "", "name": "", "roi": "", "country": "", "city": "", "color": "", "column": "", "num": "", "crop_img": "", "crop_xml": "", "crop_json": ""}]

    # pd
    data_pd = pd.read_csv(args.input_csv_path)

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

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
        crop_json_path = row['crop_json']

        plate_img_name = plate_name + '.jpg'
        plate_mask_name = plate_name + '.png'
        output_img_path = os.path.join(args.output_img_dir, plate_img_name)
        output_mask_path = os.path.join(args.output_mask_dir, plate_mask_name)
        output_mask_img_path = os.path.join(args.output_mask_img_dir, plate_mask_name)
        output_bbox_img_path = os.path.join(args.output_bbox_img_dir, plate_mask_name)

        # plate_img
        plate_img = cv2.imread(crop_img_path)

        # load object_roi_list
        object_roi_list = dataset_dict.load_object_roi(crop_xml_path, crop_json_path, args.new_style)

        # draw_mask
        draw_mask(dataset_dict, plate_img, object_roi_list, output_img_path, output_mask_path, output_mask_img_path, output_bbox_img_path)

        csv_list.append({"img_path": img_path, "json_path": json_path, "roi_img_path": output_img_path, "roi_mask_path": output_mask_path, "id": plate_id, "name": plate_name, "roi": plate_roi, "country": plate_country, "city": plate_city, "color": plate_color, "column": plate_column, "num": plate_num, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "crop_json": crop_json_path})

    # out csv
    csv_pd = pd.DataFrame(csv_list)
    csv_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="shate_20230308") 
    parser.add_argument('--seg_name', type=str, default="seg_zd_202306") 
    parser.add_argument('--input_csv_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_crop_csv/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/") 
    parser.add_argument('--new_style', action='store_true', default=False) 
    args = parser.parse_args()

    args.input_csv_path = os.path.join(args.input_csv_dir, args.date_name + '.csv')
    args.output_dir = os.path.join(args.output_dir, args.seg_name, args.date_name)

    print("gen seg mask.")
    print("date_name: {}".format(args.date_name))
    print("seg_name: {}".format(args.seg_name))
    print("input_csv_path: {}".format(args.input_csv_path))
    print("output_dir: {}".format(args.output_dir))

    ###############################################
    # dataset_zd_dict_city
    ###############################################

    args.seg_dict_name = "dataset_zd_dict_city"

    args.output_img_dir = os.path.join(args.output_dir, 'city_label/Images')
    args.output_mask_dir = os.path.join(args.output_dir, 'city_label/mask')
    args.output_mask_img_dir = os.path.join(args.output_dir, 'city_label/mask_img')
    args.output_bbox_img_dir = os.path.join(args.output_dir, 'city_label/bbox_img')
    args.output_csv_path = os.path.join(args.output_dir, 'city_label/{}.csv'.format(args.date_name))

    # 生成 seg mask
    gen_seg_mask(args)

    ###############################################
    # dataset_zd_dict_color
    ###############################################

    args.seg_dict_name = "dataset_zd_dict_color"

    args.output_img_dir = os.path.join(args.output_dir, 'color_label/Images')
    args.output_mask_dir = os.path.join(args.output_dir, 'color_label/mask')
    args.output_mask_img_dir = os.path.join(args.output_dir, 'color_label/mask_img')
    args.output_bbox_img_dir = os.path.join(args.output_dir, 'color_label/bbox_img')
    args.output_csv_path = os.path.join(args.output_dir, 'color_label/{}.csv'.format(args.date_name))

    # 生成 seg mask
    gen_seg_mask(args)