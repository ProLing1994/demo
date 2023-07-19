import argparse
import cv2
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *
from Basic.script.xml.xml_write import write_xml


def crop_image_xml(args):

    # mkdir 
    create_folder(args.output_img_dir)
    create_folder(args.output_xml_dir)

    jpg_list = np.array(os.listdir(args.input_img_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.input_xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]
    jpg_list.sort()

    for idx in tqdm(range(len(jpg_list))):

        img_path = os.path.join(args.input_img_dir, jpg_list[idx])
        xml_path = os.path.join(args.input_xml_dir, jpg_list[idx].replace(".jpg", ".xml"))

        # img
        img = cv2.imread(img_path)

        # xml
        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

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
            bndbox[0] = max(0, bndbox[0])
            bndbox[1] = max(0, bndbox[1])
            bndbox[2] = min(img.shape[1], bndbox[2])
            bndbox[3] = min(img.shape[0], bndbox[3])

            if classname not in args.crop_key_list:
                print(classname)
                continue

            crop_width = bndbox[2] - bndbox[0]
            crop_height = bndbox[3] - bndbox[1]
            crop_cener_x = (bndbox[0] + bndbox[2]) / 2
            crop_cener_y = (bndbox[1] + bndbox[3]) / 2

            if crop_width < args.crop_size[0] and crop_height < args.crop_size[1]:
            
                # crop_roi
                crop_roi = [0, 0, 0, 0]
                crop_roi[0] = crop_cener_x - args.crop_size[0] / 2
                crop_roi[1] = crop_cener_y - args.crop_size[1] / 2
                crop_roi[2] = crop_cener_x + args.crop_size[0] / 2
                crop_roi[3] = crop_cener_y + args.crop_size[1] / 2

                if crop_roi[0] < 0:
                    transform_x = 0 - crop_roi[0]
                    crop_roi[0] = 0
                    crop_roi[2] += transform_x
                if crop_roi[1] < 0:
                    transform_y = 0 - crop_roi[1]
                    crop_roi[1] = 0
                    crop_roi[3] += transform_y
                if crop_roi[2] > img.shape[1]:  
                    transform_x = crop_roi[2] - img.shape[1]
                    crop_roi[2] = img.shape[1] - 1
                    crop_roi[0] -= transform_x + 1
                if crop_roi[3] > img.shape[0]:
                    transform_y = crop_roi[3] - img.shape[0]
                    crop_roi[3] = img.shape[0] - 1
                    crop_roi[1] -= transform_y + 1

                crop_roi[0] = int(crop_roi[0] + 0.5)
                crop_roi[1] = int(crop_roi[1] + 0.5)
                crop_roi[2] = int(crop_roi[2] + 0.5)
                crop_roi[3] = int(crop_roi[3] + 0.5)

                img_name = jpg_list[idx].replace(".jpg", "")
                output_img_path = os.path.join(args.output_img_dir, img_name + '_' + classname + '.jpg')
                output_xml_path = os.path.join(args.output_xml_dir, img_name + '_' + classname + '.xml')

                # img
                img_crop = img[crop_roi[1]:crop_roi[3], crop_roi[0]:crop_roi[2]]
                cv2.imwrite(output_img_path, img_crop)

                # xml
                assert img_crop.shape[0] == 512
                assert img_crop.shape[1] == 512
                assert bndbox[0] - crop_roi[0] >= 0
                assert bndbox[1] - crop_roi[1] >= 0
                assert bndbox[2] - crop_roi[0] >= 0
                assert bndbox[3] - crop_roi[1] >= 0
                assert bndbox[0] - crop_roi[0] <= 512
                assert bndbox[1] - crop_roi[1] <= 512
                assert bndbox[2] - crop_roi[0] <= 512
                assert bndbox[3] - crop_roi[1] <= 512

                xml_bboxes = {}
                xml_bboxes[classname] = []   
                xml_bboxes[classname].append([bndbox[0] - crop_roi[0], bndbox[1] - crop_roi[1], bndbox[2] - crop_roi[0], bndbox[3] - crop_roi[1]])
                write_xml(output_xml_path, output_img_path, xml_bboxes, img_crop.shape)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Europe") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_upspeed/original/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_upspeed/crop/") 
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.date_name)
    args.output_dir = os.path.join(args.output_dir, args.date_name)

    print("crop image xml.")
    print("date_name: {}".format(args.date_name))
    print("input_dir: {}".format(args.input_dir))
    print("output_dir: {}".format(args.output_dir))

    args.input_img_dir = os.path.join(args.input_dir, 'JPEGImages')
    args.input_xml_dir = os.path.join(args.input_dir, 'Annotations')
    args.output_img_dir = os.path.join(args.output_dir, 'JPEGImages')
    args.output_xml_dir = os.path.join(args.output_dir, 'Annotations')

    args.crop_key_list = ['upspeed_spain_100', 'upspeed_spain_120', 'upspeed_spain_30', 'upspeed_spain_40', 'upspeed_spain_50', 'upspeed_spain_60', 'upspeed_spain_70', 'upspeed_spain_80', 'upspeed_spain_90', 
                          'sign_upspeed_c', 
                          'sign_height_c',
                          'sign_hand_c',
                          'sign_handb_c']
    # w, h
    args.crop_size = (512, 512)

    crop_image_xml(args)