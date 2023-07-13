import argparse
import cv2
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

# sys.path.insert(0, '/home/huanyuan/code/demo/Image')
sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *
from Basic.script.xml.xml_write import write_xml


def cal_iou_special(bbox1, bbox2):
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
    min_area = min(((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])), ((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])))
    overlaps = inters / (min_area + 1e-5)
    return overlaps


def crop_image(args):
    # mkdir
    create_folder(args.output_dir)

    # 遍历数据集
    for dataset_idx in tqdm(range(len(args.dataset_list))):
        dataset_name = args.dataset_list[dataset_idx]

        jpg_dir = os.path.join(args.input_dir, dataset_name, args.jpg_dir)
        xml_dir = os.path.join(args.input_dir, dataset_name, args.xml_dir)

        jpg_list = np.array(os.listdir(jpg_dir))
        jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
        jpg_list.sort()

        for idx in tqdm(range(len(jpg_list))):
            jpg_name = jpg_list[idx]
            jpg_path = os.path.join(jpg_dir, jpg_name)
            xml_path = os.path.join(xml_dir, jpg_name.replace(".jpg", ".xml"))

            tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
            root = tree.getroot()   # 获取根节点

            # 读取标签
            idy = 0
            for obj in root.iter('object'):
                # name
                name = obj.find('name').text.lower().strip()

                # bbox
                bbox = obj.find('bndbox')
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(float(bbox.find(pt).text)) - 1
                    bndbox.append(cur_pt)

                if name in args.select_label_list:
                    
                    # 读取图像
                    img = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
                    x1, x2 = bndbox[0], bndbox[2]
                    y1, y2 = bndbox[1], bndbox[3]
                    
                    width = x2 - x1
                    height = y2 - y1

                    center_x = int((x1 + x2)/2 + 0.5)
                    center_y = int((y1 + y2)/2 + 0.5)

                    if width < args.min_size:
                        x1 = max(0, int(center_x - args.min_size / 2))
                        x2 = min(img.shape[1], int(center_x + args.min_size / 2))

                    if height < args.min_size:
                        y1 = max(0, int(center_y - args.min_size / 2))
                        y2 = min(img.shape[0], int(center_y + args.min_size / 2))

                    crop_img = img[y1:y2, x1:x2]
                    crop_img_shape = [crop_img.shape[1], crop_img.shape[0], 3]

                    output_img_path = os.path.join(args.output_dir, dataset_name, args.jpg_dir, jpg_name.replace(".jpg", "_{}.jpg".format(idy)))
                    create_folder(os.path.dirname(output_img_path))
                    cv2.imwrite(output_img_path, crop_img)
                    
                    # xml
                    xml_bboxes = {}

                    for obj in root.iter('object'):
                        # name
                        name = obj.find('name').text.lower().strip()

                        # bbox
                        bbox = obj.find('bndbox')
                        pts = ['xmin', 'ymin', 'xmax', 'ymax']
                        bndbox = []
                        for i, pt in enumerate(pts):
                            cur_pt = int(float(bbox.find(pt).text)) - 1
                            bndbox.append(cur_pt)
                    
                        if name in args.transform_label_list:
                            
                            iou_special = cal_iou_special(bndbox, [x1, y1, x2, y2])
                            if iou_special > 0.5:

                                bndbox[0] = max(0, int(bndbox[0] - x1))
                                bndbox[2] = min(crop_img.shape[1], int(bndbox[2] - x1))
                                bndbox[1] = max(0, int(bndbox[1] - y1))
                                bndbox[3] = min(crop_img.shape[0], int(bndbox[3] - y1))
                                
                                if name not in xml_bboxes:
                                    xml_bboxes[name] = []              
                                xml_bboxes[name].append(bndbox)

                    output_xml_path = os.path.join(args.output_dir, dataset_name, args.xml_dir, jpg_name.replace(".jpg", "_{}.xml".format(idy)))
                    create_folder(os.path.dirname(output_xml_path))
                    write_xml(output_xml_path, output_img_path, xml_bboxes, crop_img_shape)

                    idy += 1


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/original"
    args.output_dir = "/yuanhuan/data/image/ZG_BMX_detection/original_crop_moto"

    args.dataset_list = ['daminghu', 'daminghu_night', 'rongheng', 'rongheng_night_hongwai']

    args.jpg_dir = "JPEGImages"
    args.xml_dir = "XML"
    args.select_label_list = ["motorcyclist", "bicyclist"]
    args.transform_label_list = ['head', 'helmet']
    
    args.min_size = 320

    crop_image(args)