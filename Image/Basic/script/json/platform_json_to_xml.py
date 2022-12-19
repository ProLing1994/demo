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

    json_list = np.array(os.listdir(args.platform_json_dir))
    json_list = json_list[[jpg.endswith('.json') for jpg in json_list]]
    json_list.sort()

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list.sort()

    for idx in tqdm(range(len(json_list))):
        # json
        json_name = json_list[idx]
        json_path = os.path.join(args.platform_json_dir, json_name)

        # jpg
        jpg_name = str(json_name).replace('.json', '.jpg')
        jpg_path = os.path.join(args.jpg_dir, jpg_name)
        assert jpg_name in jpg_list

        # xml 
        xml_name = str(jpg_name).replace('.jpg', '.xml')
        xml_path = os.path.join(args.xml_dir, xml_name)

        # read json
        with open(json_path, 'r', encoding='UTF-8') as fr:
            annotation = json.load(fr)

        ########################################################
        # 普通模式
        ########################################################
        img_width = annotation['width']
        img_height = annotation['height']
        img_shape = np.array([img_height, img_width, 3])

        xml_bboxes = {}
        for track in annotation['shapes']:
            label = track['label']
            type = track['type']

            if type == 'rectangle':
                # +1 的目的：更换了标注工具，保证 xml 结果统一：
                # ssd rfb 代码：cur_pt = int(float(bbox.find(pt).text)) - 1
                points = np.array(track['points'])
                x1 = max(int(points[0]), 1) 
                y1 = max(int(points[1]), 1) 
                x2 = min(int(points[2]), img_width - 1)
                y2 = min(int(points[3]), img_height - 1)

                if label not in xml_bboxes:
                    xml_bboxes[label] = []              
                xml_bboxes[label].append([x1, y1, x2, y2])

            elif type == 'polygon':
                points = np.array(track['points'])

                points_list = []
                for x, y in zip(points[::2], points[1::2]):
                    points_list.append([x, y])
                
                points_list = np.array(points_list)
                x1 = max(int(min(points_list[:, 0])) + 1, 1)
                y1 = max(int(min(points_list[:, 1])) + 1, 1)
                x2 = min(int(max(points_list[:, 0])) + 1, img_width - 1)
                y2 = min(int(max(points_list[:, 1])) + 1, img_height - 1)
                
                if label not in xml_bboxes:
                    xml_bboxes[label] = []              
                xml_bboxes[label].append([x1, y1, x2, y2])

        write_xml(xml_path, jpg_path, xml_bboxes, img_shape)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # # normal: UAE
    # # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_0804_0809/"
    # # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_0810_0811/"
    # # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_0828_0831/"
    # # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_0901_0903/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_0904_0905/"
    
    # args.jpg_dir = os.path.join(args.input_dir)
    # args.platform_json_dir = os.path.join(args.input_dir)
    # args.xml_dir = os.path.join(args.input_dir)
    
    # platform_json_2_xml(args)

    # normal: UAE_crop
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0804_0809/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0810_0811/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0828_0831/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0901_0903/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0904_0905/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_1024_1029/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_1/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_2/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_3/"
    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_4/"
    
    args.jpg_dir = os.path.join(args.input_dir, "Images/")
    args.platform_json_dir = os.path.join(args.input_dir, "Json/")
    args.xml_dir = os.path.join(args.input_dir, "xml/")
    
    platform_json_2_xml(args)

    # # test
    # # args.input_dir = "/yuanhuan/model/image/lpr/zd/size_signal_small/"
    # # args.input_dir = "/yuanhuan/model/image/lpr/zd/size_signal_big/"
    # # args.input_dir = "/yuanhuan/model/image/lpr/zd/size_double_small/"
    # args.input_dir = "/yuanhuan/model/image/lpr/zd/size_double_big"
    
    # args.jpg_dir = os.path.join(args.input_dir)
    # args.platform_json_dir = os.path.join(args.input_dir)
    # args.xml_dir = os.path.join(args.input_dir)

    # platform_json_2_xml(args)