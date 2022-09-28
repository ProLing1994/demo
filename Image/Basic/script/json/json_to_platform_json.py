import argparse
import json
import numpy as np
import os
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
from Basic.utils.folder_tools import *
from Basic.script.json.platform_json_write import PlatformJsonWriter


def json_2_platform_json(args):

    # mkdir 
    create_folder(args.platform_json_dir)

    json_list = np.array(os.listdir(args.json_dir))
    json_list = json_list[[jpg.endswith('.json') for jpg in json_list]]
    json_list.sort()

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]

    platform_json_writer = PlatformJsonWriter()

    for idx in tqdm(range(len(json_list))):
        # json
        json_name = json_list[idx]
        json_path = os.path.join(args.json_dir, json_name)

        # jpg
        jpg_name = str(json_name).replace('.json', '.jpg')
        assert jpg_name in jpg_list

        # read json
        with open(json_path, 'r', encoding='UTF-8') as fr:
            annotation = json.load(fr)
            
        img_width = annotation['imageWidth']
        img_height = annotation['imageHeight']

        json_bboxes = {}
        for track in annotation['shapes']:
            label = track['label']
            points = track['points']
            type = track['shape_type']
            assert type == "polygon"

            points_list = []
            for idy in range(len(points)):
                point_idx = points[idy]
                points_list.append(point_idx[0])
                points_list.append(point_idx[1])
                
            if label not in json_bboxes:
                json_bboxes[label] = []     
            json_bboxes[label].append(points_list)
        
        out_json_path = os.path.join(args.platform_json_dir, json_name)
        platform_json_writer.write_json(img_width, img_height, jpg_name, out_json_path, frame_num=idx, polygon_dict=json_bboxes)
    
    out_meta_json_path = os.path.join(args.platform_json_dir, 'meta.json')
    platform_json_writer.write_meta_json(out_meta_json_path, task_name="lpr_20220826_76174", label_list=args.label_list)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.jpg_dir = "/mnt/huanyuan/model/image/lpr/zd/seg_city_refine/"
    args.json_dir = "/mnt/huanyuan/model/image/lpr/zd/seg_city_refine/"
    args.platform_json_dir = "/mnt/huanyuan/model/image/lpr/zd/seg_city_refine_platform_json/"
    
    args.label_list = [
                        "kind", "num", 
                        "UAE", "UAE_f", 
                        "AD", "ABUDHABI", "DUBAI", "AJMAN", "SHARJAH", "SHJ", "RAK", "UMMALQAIWAIN", "FUJAIRAH",
                        "AD_f", "ABUDHABI_f", "DUBAI_f", "AJMAN_f", "SHARJAH_f", "SHJ_f", "RAK_f", "UMMALQAIWAIN_f", "FUJAIRAH_f",
                        "TAXI", "POLICE", "PUBLIC", "TRP", "PROTOCOL", "PTR", 
                        "TAXI_f", "POLICE_f", "PUBLIC_f", "TRP_f", "PROTOCOL_f", "PTR_f",
                        "YELLOW", "RED", "GREEN", "BULE", "ORANGE", "BROWN"
                        ]
    
    json_2_platform_json(args)
