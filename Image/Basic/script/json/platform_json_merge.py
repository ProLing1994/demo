import argparse
import json
import numpy as np
import os
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Image/')
# sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *
from Basic.script.json.platform_json_write import PlatformJsonWriter


def platform_json_merge(args):
    # mkdir 
    create_folder(args.output_platform_json_dir)

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]

    platform_json_writer = PlatformJsonWriter()

    for idx in tqdm(range(len(jpg_list))):
        
        # init 
        rect_list = []

        # jpg
        jpg_name = jpg_list[idx]
        # json 
        json_name = jpg_name.replace('.jpg', '.json')
        
        # platform_json
        platform_json_path = os.path.join(args.platform_json_dir, json_name)

        # read json
        with open(platform_json_path, 'r', encoding='UTF-8') as fr:
            annotation = json.load(fr)

        img_width = annotation['width']
        img_height = annotation['height']
        img_shape = np.array([img_height, img_width, 3])

        for track in annotation['shapes']:
            label = track['label']
            type = track['type']

            if type == 'rectangle':
                bndbox = []
                bndbox.extend(track['points'])
                bndbox.extend([track['label']])
                bndbox.extend(track['attributes'])
                rect_list.append(bndbox)

        # platform_json_merge
        platform_json_merge_path = os.path.join(args.platform_json_merge_dir, json_name)
        try:
            # read json
            with open(platform_json_merge_path, 'r', encoding='UTF-8') as fr:
                annotation = json.load(fr)

            img_width = annotation['width']
            img_height = annotation['height']
            img_shape = np.array([img_height, img_width, 3])

            for track in annotation['shapes']:
                label = track['label']
                type = track['type']

                if type == 'rectangle':
                    bndbox = []
                    bndbox.extend(track['points'])
                    bndbox.extend([track['label']])
                    bndbox.extend([track['attributes']])
                    rect_list.append(bndbox)
        except:
            print(platform_json_merge_path)

        out_json_path = os.path.join(args.output_platform_json_dir, json_name)
        platform_json_writer.write_json(img_width, img_height, jpg_name, out_json_path, frame_num=idx, rect_list=rect_list)

    out_meta_json_path = os.path.join(args.output_platform_json_dir, 'meta.json')
    platform_json_writer.write_meta_json(out_meta_json_path, task_name="{}_{}".format(args.task_name, len(jpg_list)), label_list=args.label_list)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/mnt/huanyuan2/data/image/LicensePlate_ocr/SHATE/2023-02-08/"
    # args.input_dir = "/mnt/huanyuan2/data/image/LicensePlate_ocr/SHATE/2023-02-09/"
    args.jpg_dir = os.path.join(args.input_dir, "JPEGImages/")
    args.platform_json_dir = os.path.join(args.input_dir, "Annotations_Json_car/")
    args.platform_json_merge_dir = os.path.join(args.input_dir, "Annotations_Json_plate/")
    args.output_platform_json_dir = os.path.join(args.input_dir, "Json/")
    args.task_name = "ST_ANPR_C27"
    args.label_list =  ['car','truck','bus','motorcycle','license']

    platform_json_merge(args)