import argparse
import cv2
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
# sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.xml.xml_write import write_xml
from Basic.utils.folder_tools import *

zg_label_dict = { "person": 'head',
                  "hat": 'helmet',
                }

ignore_label_id = [ "dog" ]

def label_to_xml(img_folder, label_folder, output_folder, args):

    img_list = os.listdir( img_folder )
    img_list = [img_name for img_name in img_list if img_name.endswith('.jpg')]
    img_list.sort()

    for img_idx in tqdm(range(len( img_list ))):
        
        # img
        img_name = img_list[img_idx]
        img_path = os.path.join(img_folder, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_shape = [img.shape[1], img.shape[0], img.shape[2]]

        # xml
        xml_name = img_name.replace('.jpg', '.xml')
        xml_path = os.path.join(label_folder, xml_name)

        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

        # 标签检测和标签转换
        xml_bboxes = {}
        for object in root.findall('object'):
            # name
            classname = str(object.find('name').text)

            # continue
            if classname in ignore_label_id:
                continue

            classname = zg_label_dict[classname]

            # bbox
            bbox = object.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                bndbox.append(cur_pt)

            if classname not in xml_bboxes:
                xml_bboxes[classname] = []

            xml_bboxes[classname].append(bndbox)
        
        xml_path = os.path.join(output_folder, img_name.replace('.jpg', '.xml'))
        write_xml(xml_path, img_path, xml_bboxes, img_shape)


def tranform(args):

    create_folder(args.output_dir)
    img_folder = os.path.join( args.input_dir, 'JPEGImages' ) 
    label_folder = os.path.join( args.input_dir, 'label' ) 

    label_to_xml(img_folder, label_folder, args.output_dir, args)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/mnt/huanyuan/temp/helmet/VOC2028/"
    args.output_dir = "/mnt/huanyuan/temp/helmet/VOC2028/XML/"

    tranform(args)