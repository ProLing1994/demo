import argparse
import cv2
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *


def intersect(box_a, box_b):
    inter_x1 = max(box_a[0], box_b[0])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y1 = max(box_a[1], box_b[1])
    inter_y2 = min(box_a[3], box_b[3])
    inter =  max(inter_x2 - inter_x1, 0.0) * max(inter_y2 - inter_y1, 0.0) 
    return inter


def check_in_crop_roi(bndbox, crop_roi):
    intersect_size = intersect(bndbox, crop_roi)
    bndbox_size = max(bndbox[2] - bndbox[0], 0.0) * max(bndbox[3] - bndbox[1], 0.0) 
    if bndbox_size == 0.0:
        return False

    intersect_iou = intersect_size / bndbox_size
    return intersect_iou > 0.3


def crop_image(args):
    # mkdir 
    create_folder(args.output_dir)
    create_folder(args.output_img_dir)
    create_folder(args.output_xml_dir)

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]
    jpg_list.sort()

    for idx in tqdm(range(len(jpg_list))):
        img_path = os.path.join(args.jpg_dir, jpg_list[idx])
        xml_path = os.path.join(args.xml_dir, jpg_list[idx].replace(".jpg", ".xml"))
        output_img_path = os.path.join(args.output_img_dir, jpg_list[idx])
        output_xml_path = os.path.join(args.output_xml_dir, jpg_list[idx].replace(".jpg", ".xml"))

        # crop_roi
        crop_roi = [0, 0, 0, 0]
        bool_find_crop_name = False
        for crop_name_idx in range(len(args.crop_name_list)):
            if np.array([True if idy in str(jpg_list[idx]) else False for idy in args.crop_name_list[crop_name_idx]]).any():
                bool_find_crop_name = True
                crop_roi = args.crop_roi_list[crop_name_idx]
        assert bool_find_crop_name == True, print(str(jpg_list[idx]))

        # img
        img = cv2.imread(img_path)
        img_crop = img[crop_roi[1]:crop_roi[3], crop_roi[0]:crop_roi[2]]
        print(img.shape, img_crop.shape)
        img_width = img_crop.shape[1]
        img_height = img_crop.shape[0]
        assert img_width == 1280
        assert img_height == 720
        cv2.imwrite(output_img_path, img_crop)

        # xml
        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点
        
        # img_width & img_height
        root.find('size').find('width').text = str(img_width)
        root.find('size').find('height').text = str(img_height)
        
        # 标签检测和标签转换
        for object in root.findall('object'):
            # name
            classname = str(object.find('name').text)

            # bbox
            bbox = object.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                bndbox.append(cur_pt)

            # check roi
            bool_in_crop_roi = check_in_crop_roi(bndbox, crop_roi)
            if bool_in_crop_roi:
                for i, pt in enumerate(pts):
                    if not i % 2:
                        bbox.find(pt).text = str(int(min(max(float(bndbox[i] - crop_roi[0]), 0,0), img_width) + 1))
                    else:
                        bbox.find(pt).text = str(int(min(max(float(bndbox[i] - crop_roi[1]), 0,0), img_height) + 1))
            else:
                object.find('name').text = 'remove'
    
        # 删除无用标签
        for object in root.findall('object'):
            classname = str(object.find('name').text)
            if classname == 'remove':
                root.remove(object)
                
        tree.write(output_xml_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ######################################
    # Annotations_HeadHelmet
    ######################################
    # ZG_BMX_detection
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/daminghu/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/daminghu_night/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/rongheng/"
    args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/rongheng_night_hongwai/"
    
    # 720p
    # # daminghu
    # args.crop_roi_list = [ [720, 430, 2000, 1150], [520, 230, 1800, 950], [410, 200, 1690, 920]]
    # args.crop_name_list = [ ['-000005'], ['-000002'], ['-000001']]

    # rongheng
    args.crop_roi_list = [ [550, 250, 1830, 970], [640, 360, 1920, 1080]]
    args.crop_name_list = [ ['-000003',], ['-000001']]

    args.jpg_dir = os.path.join(args.input_dir, 'JPEGImages') 
    args.xml_dir = os.path.join(args.input_dir, 'XML')
    args.output_dir = os.path.join(args.input_dir, 'crop_720p')
    args.output_img_dir = os.path.join(args.output_dir, 'JPEGImages')
    args.output_xml_dir = os.path.join(args.output_dir, 'XML') 

    crop_image(args)