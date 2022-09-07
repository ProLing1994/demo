import argparse
import json
import numpy as np
import os
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
# sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.xml.xml_write import write_xml


def json_xml(args):
    json_list = np.array(os.listdir(args.json_dir))
    json_list = json_list[[jpg.endswith('.json') for jpg in json_list]]

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]

    for idx in tqdm(range(len(json_list))):
        # json
        json_name = json_list[idx]
        json_path = os.path.join(args.json_dir, json_name)

        # jpg
        jpg_name = str(json_name).replace('.json', '.jpg')
        jpg_path = os.path.join(args.jpg_dir, jpg_name)
        if not jpg_name in jpg_list:
            continue
        
        # xml 
        xml_name = str(jpg_name).replace('.jpg', '.xml')
        xml_path = os.path.join(args.xml_dir, xml_name)

        # read json
        with open(json_path, 'r', encoding='UTF-8') as fr:
            try:
                annotation = json.load(fr)
            except:
                print(json_path)
                continue

        img_width = annotation['width']
        img_height = annotation['height']
        img_shape = [img_width, img_height, 3]

        xml_bboxes = {}
        for track in annotation['shapes']:
            label = track['label']
            type = track['type']

            # 目前只支持检测框(rectangle)
            if not type == 'rectangle':
                continue
            
            # +1 的目的：更换了标注工具，保证 xml 结果统一：
            # ssd rfb 代码：cur_pt = int(float(bbox.find(pt).text)) - 1
            points = np.array(track['points'])
            x1 = max(int(points[0]) + 1, 1)
            y1 = max(int(points[1]) + 1, 1)
            x2 = min(int(points[2]) + 1, img_width) 
            y2 = min(int(points[3]) + 1, img_height)

            if label not in xml_bboxes:
                xml_bboxes[label] = []              
            xml_bboxes[label].append([x1, y1, x2, y2])
    
        write_xml(xml_path, jpg_path, xml_bboxes, img_shape)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.jpg_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_json/"
    args.json_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_json/"
    args.xml_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_json/"

    json_xml(args)


    
