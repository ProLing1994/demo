import argparse
import cv2
import importlib
import numpy as np
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
from script.lpr.dataset.dataset_augment.data_augment import *


def gen_seg_mask_augument(args):
    
    # dataset_cn_dict
    dataset_dict = importlib.import_module('script.lpr.dataset.dataset_cn.dataset_dict.' + args.data_dict_name) 

    # mkdir
    create_folder(args.output_img_dir)
    create_folder(args.output_mask_dir)
    create_folder(args.output_mask_img_dir)
    create_folder(args.output_bbox_img_dir)

    # init
    csv_list = []           # [{"img_path": "", "json_path": "", "roi_img_path": "", "roi_mask_path": "", "id": "", "name": "", "roi": "", "color": "", "column": "", "num": "", "split_type": ""}]

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
        split_type = row['split_type']

        plate_img_name = plate_name + '.jpg'
        plate_mask_name = plate_name + '.png'

        # img
        img = cv2.imread(img_path)

        # plate_img
        plate_roi_list = plate_roi.split(',')
        x1 = int(plate_roi_list[0])
        x2 = int(plate_roi_list[1])
        y1 = int(plate_roi_list[2])
        y2 = int(plate_roi_list[3])
        plate_img = img[y1:y2, x1:x2]
        plate_roi = [x1, y1, x2, y2]

        # object_roi_list
        object_roi_list = []
        object_roi_list.append({"classname": plate_color, "bndbox": [0, 0, plate_img.shape[1] - 1, plate_img.shape[0] - 1]})

        # aug
        for idy in range(args.aug_times):
            
            # aug_rotate_bbox
            rotate_img, rotate_img_roi, rotate_object_roi = aug_rotate_bbox(img.copy(), plate_roi.copy(), object_roi_list.copy(), args.aug_rotate_angle_list)
            # aug_expand_bbox
            expand_img_roi, expand_object_roi = aug_expand_bbox(rotate_img, rotate_img_roi, rotate_object_roi, args.aug_expand_ratio_list)

            rotate_img_crop = rotate_img[expand_img_roi[1]:expand_img_roi[3], expand_img_roi[0]:expand_img_roi[2]]

            # color
            output_img_path = os.path.join(args.output_img_dir, plate_img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.jpg".format(idy, plate_num)))
            output_mask_path = os.path.join(args.output_mask_dir, plate_img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.png".format(idy, plate_num)))
            output_mask_img_path = os.path.join(args.output_mask_img_dir, plate_img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.png".format(idy, plate_num)))
            output_bbox_img_path = os.path.join(args.output_bbox_img_dir, plate_img_name.replace("_{}.jpg".format(plate_num), "_aug_{}_{}.png".format(idy, plate_num)))
            draw_mask(dataset_dict, rotate_img_crop, expand_object_roi, output_img_path, output_mask_path, output_mask_img_path, output_bbox_img_path)

            csv_list.append({"img_path": img_path, "json_path": json_path, "roi_img_path": output_img_path, "roi_mask_path": output_mask_path, "id": plate_id, "name": plate_name, "roi": plate_roi, "color": plate_color, "column": plate_column, "num": plate_num, "split_type": split_type})

    # out csv
    csv_pd = pd.DataFrame(csv_list)
    csv_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="shanxi_jin") 
    parser.add_argument('--seg_name', type=str, default="seg_cn_202305") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/") 
    args = parser.parse_args()

    args.data_dict_name = "dataset_cn_dict_color"
    args.output_dir = os.path.join(args.output_dir, args.seg_name, args.date_name)
    
    print("gen seg mask augment.")
    print("date_name: {}".format(args.date_name))
    print("seg_name: {}".format(args.seg_name))
    print("output_dir: {}".format(args.output_dir))

    args.output_img_dir = os.path.join(args.output_dir, 'color_label/Images_aug')
    args.output_mask_dir = os.path.join(args.output_dir, 'color_label/mask_aug')
    args.output_mask_img_dir = os.path.join(args.output_dir, 'color_label/mask_img_aug')
    args.output_bbox_img_dir = os.path.join(args.output_dir, 'color_label/bbox_img_aug')
    args.input_csv_path = os.path.join(args.output_dir, 'color_label', args.date_name + '.csv')
    args.output_csv_path = os.path.join(args.output_dir, 'color_label', args.date_name + '_augument.csv')

    args.aug_times = 5
    args.aug_rotate_angle_list = range(-20, 20)
    args.aug_expand_ratio_list = list(np.arange(0.01, 0.1, 0.01))

    gen_seg_mask_augument(args)