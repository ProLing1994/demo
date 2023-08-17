import argparse
import cv2
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *


def read_bbox(args):

    # mkdir 
    create_folder(args.output_img_dir)
    create_folder(args.output_bbox_dir)
    create_folder(args.output_mask_img_dir)

    jpg_list = np.array(os.listdir(args.input_img_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.input_xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]
    jpg_list.sort()

    for idx in tqdm(range(len(jpg_list))):
    
        img_path = os.path.join(args.input_img_dir, jpg_list[idx])
        xml_path = os.path.join(args.input_xml_dir, jpg_list[idx].replace(".jpg", ".xml"))
        output_img_path = os.path.join(args.output_img_dir, jpg_list[idx])
        output_mask_img_path = os.path.join(args.output_mask_img_dir, jpg_list[idx])
        output_bbox_path = os.path.join(args.output_bbox_dir, jpg_list[idx].replace(".jpg", ".txt"))

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
            crop_size = crop_width * crop_height
            crop_size_ratio = crop_size / (img.shape[0] * img.shape[1])
            
            # 参考官方处理代码
            if crop_size_ratio > 0.8 or crop_size_ratio < 0.02:
                continue

            # resize
            width = img.shape[1]
            height = img.shape[0]
            reszie_ratio = (args.data_size[0] / (height), args.data_size[1] / (width))

            img = cv2.resize(img, (args.data_size[0], args.data_size[1]), interpolation=cv2.INTER_CUBIC)
            bndbox[0] = int(bndbox[0] * reszie_ratio[0])
            bndbox[1] = int(bndbox[1] * reszie_ratio[1])
            bndbox[2] = int(bndbox[2] * reszie_ratio[0])
            bndbox[3] = int(bndbox[3] * reszie_ratio[1])

            # img
            cv2.imwrite(output_img_path, img)
            
            # mask_img
            mask_img = np.zeros(img.shape, dtype=img.dtype)
            contours = []
            contours.append([bndbox[0], bndbox[1]])
            contours.append([bndbox[2], bndbox[1]])
            contours.append([bndbox[2], bndbox[3]])
            contours.append([bndbox[0], bndbox[3]])
            contours = [np.array(contours).reshape(-1, 1, 2)]
            mask_img = cv2.drawContours(mask_img, contours, -1, (0, 0, 255), cv2.FILLED)
            mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.3, gamma=0.)
            cv2.imwrite(output_mask_img_path, mask_img)
            
            # bbox
            f = open(output_bbox_path, 'w')
            print(bndbox[0], bndbox[1], bndbox[2], bndbox[3], file=f)
            f.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="20230616") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/Taxi_remnant/sd_crop_0810/shenzhen/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/Taxi_remnant/sd_training_0810/shenzhen/") 
    parser.add_argument('--crop_key_list', type=list, default=['bottle',]) 
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.date_name)
    args.output_dir = os.path.join(args.output_dir, args.date_name)
    args.data_size = (512, 512)

    print("read bbox.")
    print("date_name: {}".format(args.date_name))
    print("input_dir: {}".format(args.input_dir))
    print("output_dir: {}".format(args.output_dir))

    args.input_img_dir = os.path.join(args.input_dir, 'JPEGImages')
    args.input_xml_dir = os.path.join(args.input_dir, 'Annotations')
    args.output_img_dir = os.path.join(args.output_dir, 'JPEGImages')
    args.output_bbox_dir = os.path.join(args.output_dir, 'bboxes')
    args.output_mask_img_dir = os.path.join(args.output_dir, 'mask_imgs')

    print(args.crop_key_list)
    read_bbox(args)