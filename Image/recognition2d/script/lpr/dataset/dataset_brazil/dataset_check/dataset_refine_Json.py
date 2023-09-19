import argparse
import io
import json
import os
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Basic.script.json.platform_json_write import PlatformJsonWriter

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from Image.recognition2d.script.lpr.dataset.dataset_brazil.dataset_dict.dataset_brazil_dict_normal import *

def cal_iou(bbox1, bbox2):
    ixmin = max(bbox1[0], bbox2[0])
    iymin = max(bbox1[1], bbox2[1])
    ixmax = min(bbox1[2], bbox2[2])
    iymax = min(bbox1[3], bbox2[3])
    iw = max(ixmax - ixmin, 0.)
    ih = max(iymax - iymin, 0.)
    inters = iw * ih
    uni = ((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) + \
            (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) - \
            inters)
    overlaps = inters / uni
    return overlaps


def find_refine_jpg(x1, x2, y1, y2, refine_data_json):

    find_bool = False
    find_name = ""
    bbox1 = []
    bbox1.append(x1)
    bbox1.append(y1)
    bbox1.append(x2)
    bbox1.append(y2)

    for cell in refine_data_json['shapes']:

        # plate num
        plate_num = json_load_object_plate_num(cell)

        # plate roi
        refine_x1, refine_x2, refine_y1, refine_y2, w, h = json_load_object_plate_points(cell)

        bbox2 = []
        bbox2.append(refine_x1)
        bbox2.append(refine_y1)
        bbox2.append(refine_x2)
        bbox2.append(refine_y2)
        overlaps = cal_iou(bbox1, bbox2)
        
        if overlaps > 0.8:
            find_bool = True
            find_name = plate_num
            break

    return find_bool, find_name       


def dataset_refine(args):

    # mkdir
    create_folder(args.json_refine_dir)

    # img list 
    img_list = get_sub_filepaths_suffix(args.img_dir, ".jpg")
    img_list.sort()
    print("data len: ", len(img_list))

    # refine json list 
    refine_json_list = os.listdir(args.refine_dir)
    print("refine len: ", len(refine_json_list))
    refine_num = 0

    for idx in tqdm(range(len(img_list))):

        img_name = os.path.basename(img_list[idx]).replace(".jpg", "")
        img_path = img_list[idx]
        json_name = img_name + ".json"
        json_path = os.path.join(args.json_dir, json_name)
        refine_json_path = os.path.join(args.json_refine_dir, json_name)
        refine_path = os.path.join(args.refine_dir, json_name)

        # json 
        with io.open(json_path, "r", encoding="UTF-8") as f:
            data_json = json.load(f, encoding='utf-8')
            f.close()
        
        if json_name in refine_json_list:

            # json 
            with io.open(refine_path, "r", encoding="UTF-8") as f:
                refine_data_json = json.load(f, encoding='utf-8')
                f.close()

            for cell in data_json['shapes']:
    
                # plate num
                plate_num = json_load_object_plate_num(cell)
                if 'none' in plate_num.lower() or plate_num.lower() == '':
                    continue

                # plate roi
                try:
                    x1, x2, y1, y2, w, h = json_load_object_plate_points(cell)
                except:
                    print('"img_path": {}, "json_path": {},"type": "plate_roi"'.format(img_path, json_path))
                    continue
                
                # plate status
                plate_status = json_load_object_plate_status(cell)
                if plate_status != status_name_list[0]:
                    continue

                # plate column
                plate_column, load_plate_column = json_load_object_plate_column(cell, w, h)
                if plate_column == column_name_list[0]:
                    print('"img_path": {}, "json_path": {},"type": "plate_column", "value": {}'.format(img_path, json_path, load_plate_column))
                    continue
                
                # ('cover-motorplate', 2905), ('cover-plaet', 37229), ('fuzzy-motorplate', 22150), ('fuzzy-plate', 84662), ('lince-motorplate', 19846), ('lince-plate', 113262)
                find_bool, find_name = find_refine_jpg(x1, x2, y1, y2, refine_data_json)
                if find_bool:
                    for json_attributes in cell["attributes"]:
                        if json_attributes["name"] == "id":
                            if json_attributes["value"] != find_name:
                                print("{}, {}->{}".format(refine_num, plate_num, find_name))
                                json_attributes["value"] = find_name
                        else:
                            raise Exception

            refine_num += 1

        else:
            pass

        with io.open(refine_json_path, "w", encoding="UTF-8") as f:
            json.dump(data_json, f)
            f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="brazil_02210_202301") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil/") 
    parser.add_argument('--refine_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/refine/20230817_698/json_v0/") 
    parser.add_argument('--img_folder', type=str, default="JPEGImages") 
    parser.add_argument('--json_folder', type=str, default="Json_refine") 
    parser.add_argument('--json_refine_folder', type=str, default="Json_refine_2") 
    args = parser.parse_args()

    args.img_dir = os.path.join(args.input_dir, args.date_name, args.img_folder)
    args.json_dir = os.path.join(args.input_dir, args.date_name, args.json_folder)
    args.json_refine_dir = os.path.join(args.input_dir, args.date_name, args.json_refine_folder)

    print("dataset refine.")
    print("date_name: {}".format(args.date_name))
    print("input_dir: {}".format(args.img_dir))
    
    # 生成 dataset refine
    dataset_refine(args)