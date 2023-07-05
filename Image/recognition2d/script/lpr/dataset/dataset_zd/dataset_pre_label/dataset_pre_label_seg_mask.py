import argparse
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Basic.script.json.platform_json_write import PlatformJsonWriter

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from infer.lpr_seg import LPRSegCaffe, LPRSegPytorch


def pre_label(args):

    # mkdir 
    create_folder(args.output_json_dir)

    # init 
    if args.caffe_bool:
        city_seg = LPRSegCaffe(args.city_seg_caffe_prototxt, args.city_seg_caffe_model_path, args.city_seg_dict_name)
        color_seg = LPRSegCaffe(args.color_seg_caffe_prototxt, args.color_seg_caffe_model_path, args.color_seg_dict_name)
    elif args.pytorch_bool:
        city_seg = LPRSegPytorch(args.city_seg_pth_path, args.city_seg_dict_name)
        color_seg = LPRSegPytorch(args.color_seg_pth_path, args.color_seg_dict_name)
    
    # img list
    jpg_list = np.array(os.listdir(args.input_img_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list.sort()

    platform_json_writer = PlatformJsonWriter()

    for idx in tqdm(range(len(jpg_list))):

        img_name = jpg_list[idx]
        img_path = os.path.join(args.input_img_dir, img_name)
        tqdm.write(img_path)

        img = cv2.imread(img_path) 

        # info 
        image_width = img.shape[1]
        image_height = img.shape[0]

        # run
        city_preds_mask, city_seg_mask, city_seg_bbox, city_seg_info = city_seg.run(img)
        color_preds_mask, color_seg_mask, color_seg_bbox, color_seg_info = color_seg.run(img)

        seg_bbox = {}
        seg_bbox.update(city_seg_bbox)
        seg_bbox.update(color_seg_bbox)

        # 标签检测和标签转换
        rect_list = []
        for key in seg_bbox.keys():
            for idy in range(len(seg_bbox[key])):
                rect_list.append([seg_bbox[key][idy][0], seg_bbox[key][idy][1], seg_bbox[key][idy][0] + seg_bbox[key][idy][2], seg_bbox[key][idy][1] + seg_bbox[key][idy][3], key])

        # json
        output_json_path = os.path.join(args.output_json_dir, str(img_name).replace('.jpg', '.json'))
        platform_json_writer.write_json(image_width, image_height, img_name, output_json_path, frame_num=idx, rect_list=rect_list)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="shate_20230308") 
    parser.add_argument('--input_crop_data_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_crop_new/") 
    parser.add_argument('--json_name', type=str, default="Jsons_Prelabel") 
    args = parser.parse_args()

    args.caffe_bool = False
    args.pytorch_bool = True

    # zd: seg_city_cartype_kind_num_zd_1220
    args.city_seg_caffe_prototxt = ""
    args.city_seg_caffe_model_path = ""
    args.city_seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_1220/LaneNetNova.pth"
    args.city_seg_dict_name = "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_city_2022"

    # zd: seg_color_zd_1220
    args.color_seg_caffe_prototxt = ""
    args.color_seg_caffe_model_path = ""
    args.color_seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_color_zd_1220/LaneNetNova.pth"
    args.color_seg_dict_name = "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_color"

    args.input_dir = os.path.join(args.input_crop_data_dir, args.date_name)
    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.output_json_dir = os.path.join(args.input_dir, args.json_name)

    print("dataset pre label.")
    print("date_name: {}".format(args.date_name))
    print("input_img_dir: {}".format(args.input_img_dir))
    print("output_json_dir: {}".format(args.output_json_dir))

    pre_label(args)