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


def json_2_platform_json(args):

    # init
    platform_json_writer = PlatformJsonWriter()

    # mkdir
    create_folder(args.platform_json_dir)

    # img list 
    img_list = get_sub_filepaths_suffix(args.img_dir, ".jpg")
    img_list.sort()
    print("data len: ", len(img_list))

    for idx in tqdm(range(len(img_list))):

        img_name = os.path.basename(img_list[idx]).replace(".jpg", "")
        img_path = img_list[idx]

        json_name = img_name + ".json"
        json_path = os.path.join(args.json_dir, json_name)
        out_json_path = os.path.join(args.platform_json_dir, json_name)

        # init 
        rect_list = []

        # json
        with io.open(json_path, "r", encoding="UTF-8") as f:
            data_json = json.load(f, encoding='utf-8')
            f.close()

        for cell in data_json['shapes']:

            label = cell["label"]

            # plate points
            x1, x2, y1, y2, w, h = json_load_object_plate_points(cell)

            if label in ['bus', 'car', 'truck']:
                rect_list.append([x1, y1, x2, y2, label])
            elif label in ['bicyclist']:
                rect_list.append([x1, y1, x2, y2, "bicycle"])
            elif label in ['motorcyclist', 'motorcycle']:
                rect_list.append([x1, y1, x2, y2, "motorcycle"])
            elif label in ['kind', 'num']:
                continue
            else:
                # plate num
                plate_num = json_load_object_plate_num(cell)

                # plate color
                plate_color, _ = json_load_object_plate_color(cell)
                if plate_color == color_name_list[0]:
                    plate_color = "unknown"

                # plate column
                plate_column, _ = json_load_object_plate_column(cell, w, h)

                # plate status
                plate_status = json_load_object_plate_status(cell)

                if label in ['cover-motorplate', 'fuzzy-motorplate', 'lince-motorplate']:

                    rect_list.append([x1, y1, x2, y2, "license", [{"name": "number", "value": plate_num}, {"name": "color", "value": plate_color}, {"name": "column", "value": plate_column}, {"name": "type", "value": "motorcycle"}, {"name": "status", "value": plate_status}]])

                elif label in ['cover-plaet', 'fuzzy-plate', 'lince-plate']:

                    rect_list.append([x1, y1, x2, y2, "license", [{"name": "number", "value": plate_num}, {"name": "color", "value": plate_color}, {"name": "column", "value": plate_column}, {"name": "type", "value": "car"}, {"name": "status", "value": plate_status}]])
        
        platform_json_writer.write_json(2592, 1920, img_name, out_json_path, frame_num=idx, rect_list=rect_list)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="brazil_02210_202301") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil/") 
    parser.add_argument('--img_folder', type=str, default="JPEGImages") 
    parser.add_argument('--json_folder', type=str, default="Json_refine_2") 
    parser.add_argument('--platform_json_folder', type=str, default="Json_Plateform") 
    args = parser.parse_args()

    args.img_dir = os.path.join(args.input_dir, args.date_name, args.img_folder)
    args.json_dir = os.path.join(args.input_dir, args.date_name, args.json_folder)
    args.platform_json_dir = os.path.join(args.input_dir, args.date_name, args.platform_json_folder)

    print("dataset csv.")
    print("date_name: {}".format(args.date_name))
    print("input_dir: {}".format(args.img_dir))
    
    json_2_platform_json(args)