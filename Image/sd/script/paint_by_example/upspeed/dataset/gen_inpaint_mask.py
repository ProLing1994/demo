import argparse
import cv2
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *


def gen_inpaint_mask(args):

    # mkdir
    create_folder(args.output_mask_dir)
    create_folder(args.output_reference_dir)

    jpg_list = np.array(os.listdir(args.input_img_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.input_xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]
    jpg_list.sort()

    for idx in tqdm(range(len(jpg_list))):

        img_path = os.path.join(args.input_img_dir, jpg_list[idx])
        xml_path = os.path.join(args.input_xml_dir, jpg_list[idx].replace(".jpg", ".xml"))
        output_mask_path = os.path.join(args.output_mask_dir, jpg_list[idx])
        output_reference_path = os.path.join(args.output_reference_dir, jpg_list[idx])

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

            # mask
            mask = np.zeros(img.shape, dtype=img.dtype)
            contours = []
            contours.append([bndbox[0], bndbox[1]])
            contours.append([bndbox[2], bndbox[1]])
            contours.append([bndbox[2], bndbox[3]])
            contours.append([bndbox[0], bndbox[3]])
            contours = [np.array(contours).reshape(-1, 1, 2)]
            mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)
            cv2.imwrite(output_mask_path, mask)

            # reference
            # crop_img = img[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]]
            crop_img = img[max(0, bndbox[1]-args.crop_expand_pixel):min(bndbox[3]+args.crop_expand_pixel, img.shape[0]), max(0, bndbox[0]-args.crop_expand_pixel):min(bndbox[2]+args.crop_expand_pixel, img.shape[1])]
            cv2.imwrite(output_reference_path, crop_img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Europe") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_upspeed/crop/") 
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.date_name)

    print("gen inpaint mask.")
    print("date_name: {}".format(args.date_name))
    print("input_dir: {}".format(args.input_dir))

    args.input_img_dir = os.path.join(args.input_dir, 'JPEGImages')
    args.input_xml_dir = os.path.join(args.input_dir, 'Annotations')
    args.output_mask_dir = os.path.join(args.input_dir, 'masks')
    args.output_reference_dir = os.path.join(args.input_dir, 'references')

    args.crop_key_list = ['upspeed_spain_100', 'upspeed_spain_120', 'upspeed_spain_30', 'upspeed_spain_40', 'upspeed_spain_50', 'upspeed_spain_60', 'upspeed_spain_70', 'upspeed_spain_80', 'upspeed_spain_90', 
                          'sign_upspeed_c', 
                          'sign_height_c',
                          'sign_hand_c',
                          'sign_handb_c']
    args.crop_expand_pixel = 30
    gen_inpaint_mask(args)