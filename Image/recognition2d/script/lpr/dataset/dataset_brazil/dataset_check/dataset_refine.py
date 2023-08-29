import argparse
import cv2
import io
import json
import os
import pandas as pd
import shutil
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Basic.script.json.platform_json_write import PlatformJsonWriter

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from Image.recognition2d.script.lpr.dataset.dataset_brazil.dataset_dict.dataset_brazil_dict_normal import *


def find_refine_jpg(plate_prefix_name, refine_list):

    find_bool = False
    find_name = ""

    for idx in range(len(refine_list)):
        refine_name = os.path.basename(refine_list[idx])
        if plate_prefix_name in refine_name:
            find_bool = True
            find_name = refine_name
            break
    return find_bool, find_name                       


def dataset_refine(args):

    # mkdir
    create_folder(args.json_refine_dir)

    # img list 
    img_list = get_sub_filepaths_suffix(args.img_dir, ".jpg")
    img_list.sort()
    print("data len: ", len(img_list))

    # refine list 
    refine_list = get_sub_filepaths_suffix(args.refine_dir, ".jpg")
    print("refine len: ", len(refine_list))
    refine_num = 0

    for idx in tqdm(range(len(img_list))):

        plate_id = 0
        img_name = os.path.basename(img_list[idx]).replace(".jpg", "")
        img_path = img_list[idx]
        json_name = img_name + ".json"
        json_path = os.path.join(args.json_dir, json_name)
        refine_json_path = os.path.join(args.json_refine_dir, json_name)

        # json 
        try:
            with io.open(json_path, "r", encoding="UTF-8") as f:
                data_json = json.load(f, encoding='utf-8')
                f.close()
        except:
            continue
        
        for cell in data_json['shapes']:

            # plate num
            plate_num = json_load_object_plate_num(cell)
            if 'none' in plate_num.lower() or plate_num.lower() == '':
                continue

            # plate roi
            try:
                x1, x2, y1, y2, w, h = json_load_object_plate_points(cell)
                plate_roi = str("{},{},{},{}".format(x1, x2, y1, y2))
            except:
                print('"img_path": {}, "json_path": {},"type": "plate_roi"'.format(img_path, json_path))
                continue
            
            # plate status
            plate_status = json_load_object_plate_status(cell)
            if plate_status != status_name_list[0]:
                continue

            # plate color
            plate_color, load_plate_color = json_load_object_plate_color(cell)

            # plate column
            plate_column, load_plate_column = json_load_object_plate_column(cell, w, h)
            if plate_column == column_name_list[0]:
                print('"img_path": {}, "json_path": {},"type": "plate_column", "value": {}'.format(img_path, json_path, load_plate_column))
                continue

            # plate num check
            # 普通车牌：7
            if not ((len(plate_num) == 7)):
                print('"img_path": {}, "json_path": {},"type": "plate_num", "value": {}'.format(img_path, json_path, plate_num))
                continue
            
            # 普通车牌：前三位必须为字母
            bool_error_plate_num = np.array([True if str_num not in kind_num_labels[11:] else False for str_num in plate_num[:3]]).sum()
            if bool_error_plate_num:
                print('"img_path": {}, "json_path": {},"type": "plate_num", "value": {}'.format(img_path, json_path, plate_num))
                continue  
            
            # 普通车牌：第四、六、七位必须为数字
            bool_error_plate_num = np.array([True if str_num not in kind_num_labels[:11] else False for str_num in plate_num[3]]).sum()
            bool_error_plate_num += np.array([True if str_num not in kind_num_labels[:11] else False for str_num in plate_num[5]]).sum()
            bool_error_plate_num += np.array([True if str_num not in kind_num_labels[:11] else False for str_num in plate_num[6]]).sum()
            if bool_error_plate_num:
                print('"img_path": {}, "json_path": {},"type": "plate_num", "value": {}'.format(img_path, json_path, plate_num))
                continue  
            
            # 普通车牌：第五位可以为数字和字母
            # 当前阶段，如果是 1、0、I、O 进行检修
            bool_error_plate_num = np.array([True if str_num in ['1', '0', 'I', 'O'] else False for str_num in plate_num[4]]).sum()
            if bool_error_plate_num:
                print('"img_path": {}, "json_path": {},"type": "plate_num", "value": {}'.format(img_path, json_path, plate_num))
                continue  

            plate_name = args.data_format.format(img_name, plate_id, plate_color, plate_column, plate_num)
            plate_prefix_name = args.data_prefix_format.format(img_name, plate_id)
            plate_id += 1

            find_bool, find_name = find_refine_jpg(plate_prefix_name, refine_list)
            find_plate_name = find_name.split('_')[-1].split('.')[0]
            if find_bool:
                for json_attributes in cell["attributes"]:
                    if json_attributes["name"] == "id":
                        json_attributes["value"] = find_plate_name
                    else:
                        raise Exception
                refine_num += 1
                print("{}, {}->{}".format(refine_num, plate_num, find_plate_name))

        with io.open(refine_json_path, "w", encoding="UTF-8") as f:
            json.dump(data_json, f)
            f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Brazil_new_style") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil/") 
    parser.add_argument('--refine_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/refine") 
    parser.add_argument('--img_folder', type=str, default="JPEGImages") 
    parser.add_argument('--json_folder', type=str, default="Json") 
    parser.add_argument('--json_refine_folder', type=str, default="Json_refine") 
    args = parser.parse_args()

    args.data_format = "{}-{:0>2d}_{}_{}_{}"        # name-id_颜色_单双行_车牌号
    args.data_prefix_format = "{}-{:0>2d}"          # name-id
    args.img_dir = os.path.join(args.input_dir, args.date_name, args.img_folder)
    args.json_dir = os.path.join(args.input_dir, args.date_name, args.json_folder)
    args.json_refine_dir = os.path.join(args.input_dir, args.date_name, args.json_refine_folder)

    print("dataset refine.")
    print("date_name: {}".format(args.date_name))
    print("input_dir: {}".format(args.img_dir))
    
    # 生成 dataset refine
    dataset_refine(args)