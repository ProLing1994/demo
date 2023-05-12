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
    
    # dataset_cn_dict
    dataset_dict = importlib.import_module('script.lpr.dataset.dataset_cn.dataset_dict.' + args.seg_dict_name) 

    # mkdir
    create_folder(args.output_img_dir)
    create_folder(args.output_mask_dir)
    create_folder(args.output_mask_img_dir)
    create_folder(args.output_bbox_img_dir)

    # init 
    csv_list = []           # [{"img_path": "", "json_path": "", "roi_img_path": "", "roi_mask_path": "", "id": "", "name": "", "roi": "", "color": "", "column": "", "num": ""}]

    # pd
    data_pd = pd.read_csv(args.input_csv_path)

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        # info
        img_path = row['img_path']
        json_path = row['json_path']
        plate_id = row['id'] 
        plate_name = row['name']
        plate_roi = row['roi'] 
        plate_color = row['color']
        plate_column = row['column']
        plate_num = row['num']

        plate_img_name = plate_name + '.jpg'
        plate_mask_name = plate_name + '.png'
        output_img_path = os.path.join(args.output_img_dir, plate_img_name)
        output_mask_path = os.path.join(args.output_mask_dir, plate_mask_name)
        output_mask_img_path = os.path.join(args.output_mask_img_dir, plate_mask_name)
        output_bbox_img_path = os.path.join(args.output_bbox_img_dir, plate_mask_name)

        # img
        img = cv2.imread(img_path)

        # plate_img
        plate_roi_list = plate_roi.split(',')
        x1 = int(plate_roi_list[0])
        x2 = int(plate_roi_list[1])
        y1 = int(plate_roi_list[2])
        y2 = int(plate_roi_list[3])
        plate_img = img[y1:y2, x1:x2]

        # object_roi_list
        object_roi_list = []
        object_roi_list.append({"classname": plate_color, "bndbox": [0, 0, plate_img.shape[1] - 1, plate_img.shape[0] - 1]})

        draw_mask(dataset_dict, plate_img, object_roi_list, output_img_path, output_mask_path, output_mask_img_path, output_bbox_img_path)

        csv_list.append({"img_path": img_path, "json_path": json_path, "roi_img_path": output_img_path, "roi_mask_path": output_mask_path, "id": plate_id, "name": plate_name, "roi": plate_roi, "color": plate_color, "column": plate_column, "num": plate_num})
        
    # out csv
    csv_pd = pd.DataFrame(csv_list)
    csv_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")

        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="shanxi_jin") 
    parser.add_argument('--seg_name', type=str, default="seg_cn_202305") 
    parser.add_argument('--input_csv_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/cn/china_csv/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/") 
    args = parser.parse_args()

    args.seg_dict_name = "dataset_cn_dict_color"
    args.input_csv_path = os.path.join(args.input_csv_dir, args.date_name + '.csv')
    args.output_dir = os.path.join(args.output_dir, args.seg_name, args.date_name)

    print("gen seg mask.")
    print("date_name: {}".format(args.date_name))
    print("seg_name: {}".format(args.seg_name))
    print("input_csv_path: {}".format(args.input_csv_path))
    print("output_dir: {}".format(args.output_dir))

    args.output_img_dir = os.path.join(args.output_dir, 'color_label/Images')
    args.output_mask_dir = os.path.join(args.output_dir, 'color_label/mask')
    args.output_mask_img_dir = os.path.join(args.output_dir, 'color_label/mask_img')
    args.output_bbox_img_dir = os.path.join(args.output_dir, 'color_label/bbox_img')
    args.output_csv_path = os.path.join(args.output_dir, 'color_label/{}.csv'.format(args.date_name))

    # 生成 seg mask
    gen_seg_mask(args)
