import argparse
import cv2
import json
import numpy as np
import os
import sys 
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo/Image')
sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *
from Basic.script.xml.xml_write import write_xml


def platform_json_2_xml(args):

    # mkdir 
    create_folder(args.xml_dir)

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list.sort()

    json_img_list = np.array(os.listdir(args.json_img_dir))
    json_img_list = json_img_list[[jpg.endswith('.jpg') for jpg in json_img_list]]
    json_img_list.sort()

    assert len(jpg_list) == len(json_img_list)

    json_list = np.array(os.listdir(args.platform_json_dir))
    json_list = json_list[[jpg.endswith('.json') for jpg in json_list]]
    json_list.sort()


    for idx in tqdm(range(len(jpg_list))):
        # jpg 
        jpg_name = jpg_list[idx]
        jpg_path = os.path.join(args.jpg_dir, jpg_name)

        # json img
        json_img_name = json_img_list[idx]
        json_img_path = os.path.join(args.json_img_dir, json_img_name)

        # json 
        json_name = json_img_name.replace('.jpg', '.json')
        json_path = os.path.join(args.platform_json_dir, json_name)
        
        # xml 
        xml_name = str(jpg_name).replace('.jpg', '.xml')
        xml_path = os.path.join(args.xml_dir, xml_name)

        if not os.path.exists(json_path):
            print(json_path)
            continue

        # read json
        with open(json_path, 'r', encoding='UTF-8') as fr:
            annotation = json.load(fr)

        #######################################################
        # 对齐数据（临时需求）
        #######################################################

        img = cv2.imread(jpg_path)
        img_shape = img.shape
        img_width = img.shape[1]
        img_height = img.shape[0]

        # 对齐数据（临时需求）
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = img_gray[0: int(img_height * 3/4), 0: int(img_width * 3/4)]
        json_img = cv2.imread(json_img_path, 0)
        res = cv2.matchTemplate(json_img, img_gray, cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where(res >= threshold)
        y_shift = loc[0][0]
        x_shift = loc[1][0]
        
        xml_bboxes = {}
        for track in annotation['shapes']:
            label = track['label']
            type = track['type']

            if type == 'rectangle':
                # +1 的目的：更换了标注工具，保证 xml 结果统一：
                # ssd rfb 代码：cur_pt = int(float(bbox.find(pt).text)) - 1
                points = np.array(track['points'])
                x1 = max(int(points[0]) + 1 - x_shift, 1) 
                y1 = max(int(points[1]) + 1 - y_shift, 1) 
                x2 = min(int(points[2]) + 1 - x_shift, img_width - 1)
                y2 = min(int(points[3]) + 1 - y_shift, img_height - 1)
                w = x2 - x1
                h = y2 - y1
                if w < 5 or h < 5:
                    continue

                if label not in xml_bboxes:
                    xml_bboxes[label] = []              
                xml_bboxes[label].append([x1, y1, x2, y2])

            elif type == 'polygon':
                points = np.array(track['points'])

                points_list = []
                for x, y in zip(points[::2], points[1::2]):
                    points_list.append([x, y])
                
                points_list = np.array(points_list)
                x1 = max(int(min(points_list[:, 0])) + 1 - x_shift, 1)
                y1 = max(int(min(points_list[:, 1])) + 1 - y_shift, 1)
                x2 = min(int(max(points_list[:, 0])) + 1 - x_shift, img_width - 1)
                y2 = min(int(max(points_list[:, 1])) + 1 - y_shift, img_height - 1)
                w = x2 - x1
                h = y2 - y1
                if w < 5 or h < 5:
                    continue
                
                if label not in xml_bboxes:
                    xml_bboxes[label] = []              
                xml_bboxes[label].append([x1, y1, x2, y2])

        write_xml(xml_path, jpg_path, xml_bboxes, img_shape)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0810_0811/"
    args.jpg_dir = os.path.join(args.input_dir, "Images/")
    args.platform_json_dir = os.path.join(args.input_dir, "Json/")
    args.xml_dir = os.path.join(args.input_dir, "xml/")
    args.json_img_dir = os.path.join(args.input_dir, "Json_Img_temp/")

    platform_json_2_xml(args)