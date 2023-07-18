import argparse
import json
import numpy as np
import os
import sys 
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Basic.script.json.json_write import write_json


def platform_json_2_json(args):

    # mkdir 
    create_folder(args.json_dir)

    json_list = np.array(os.listdir(args.platform_json_dir))
    json_list = json_list[[jpg.endswith('.json') for jpg in json_list]]
    json_list.sort()

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]

    for idx in tqdm(range(len(json_list))):
        # json
        json_name = json_list[idx]
        json_path = os.path.join(args.platform_json_dir, json_name)

        # jpg
        jpg_name = str(json_name).replace('.json', '.jpg')
        if jpg_name not in jpg_list:
            continue

        # read json
        with open(json_path, 'r', encoding='UTF-8') as fr:
            annotation = json.load(fr)

        img_width = annotation['width']
        img_height = annotation['height']
        image_shape = np.array([img_height, img_width])

        json_bboxes = {}
        for track in annotation['shapes']:
            label = track['label']
            points = track['points']
            type = track['type']
            
            if type != args.type:
                continue

            points_list = []
            for x, y in zip(points[::2], points[1::2]):
                points_list.append([x, y])

            if label not in json_bboxes:
                json_bboxes[label] = []     
            json_bboxes[label].append(points_list)

        output_json_path = os.path.join(args.json_dir, json_name)
        write_json(output_json_path, jpg_name, image_shape, json_bboxes, "polygon")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.jpg_dir = "/yuanhuan/data/image/RM_C28_safeisland/original/test/JPEGImages/"
    args.platform_json_dir = "/yuanhuan/data/image/RM_C28_safeisland/original/test/Jsons/"
    args.json_dir = "/yuanhuan/data/image/RM_C28_safeisland/original/test/Json/"
    args.type = "polygon"
    # args.type = "polyline"
    platform_json_2_json(args)
