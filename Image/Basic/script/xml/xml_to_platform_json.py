import argparse
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *
from Basic.script.json.platform_json_write import PlatformJsonWriter


def xml_2_platform_json(args):
    
    # mkdir 
    create_folder(args.platform_json_dir)

    xml_list = np.array(os.listdir(args.xml_dir))
    xml_list = xml_list[[xml.endswith('.xml') for xml in xml_list]]
    xml_list.sort()

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    
    platform_json_writer = PlatformJsonWriter()

    for idx in tqdm(range(len(xml_list))):
        # xml
        xml_name = xml_list[idx]
        xml_path = os.path.join(args.xml_dir, xml_name)

        # jpg
        jpg_name = str(xml_name).replace('.xml', '.jpg')
        assert jpg_name in jpg_list

        # read xml
        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

        # img_width & img_height
        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)

        # 标签检测和标签转换
        rect_list = []
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
            
            bndbox.extend([classname])
            rect_list.append(bndbox)
        
        json_name = str(xml_name).replace('.xml', '.json')
        out_json_path = os.path.join(args.platform_json_dir, json_name)
        platform_json_writer.write_json(img_width, img_height, jpg_name, out_json_path, frame_num=idx, rect_list=rect_list)

    out_meta_json_path = os.path.join(args.platform_json_dir, 'meta.json')
    platform_json_writer.write_meta_json(out_meta_json_path, task_name="{}_{}".format(args.task_name, len(xml_list)), label_list=args.label_list)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # anpr
    # # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0804_0809/"
    # # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0810_0811/"
    # # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0828_0831/"
    # # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0901_0903/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0904_0905/"
    # args.jpg_dir = os.path.join(args.input_dir, "Images/")
    # args.xml_dir = os.path.join(args.input_dir, "xml/")
    # args.platform_json_dir = os.path.join(args.input_dir, "Json/")
    # args.task_name = "anpr"
    # args.label_list =  [
    #                     "kind", "num", 
    #                     "UAE", "UAE_f", "Oman", "Oman_f", 
    #                     "AD", "ABUDHABI", "DUBAI", "AJMAN", "SHARJAH", "SHJ", "RAK", "UMMALQAIWAIN", "FUJAIRAH",
    #                     "AD_f", "ABUDHABI_f", "DUBAI_f", "AJMAN_f", "SHARJAH_f", "SHJ_f", "RAK_f", "UMMALQAIWAIN_f", "FUJAIRAH_f",
    #                     "TAXI", "POLICE", "PUBLIC", "TRP", "PROTOCOL", "PTR", 
    #                     "TAXI_f", "POLICE_f", "PUBLIC_f", "TRP_f", "PROTOCOL_f", "PTR_f",
    #                     "YELLOW", "RED", "GREEN", "BULE", "ORANGE", "BROWN"
    #                     ]
    
    # BM_ANPR_c27
    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil/Brazil_src2/groundTruth/"
    args.jpg_dir = os.path.join(args.input_dir, "JPEGImages/")
    args.xml_dir = os.path.join(args.input_dir, "Annotations/")
    args.platform_json_dir = os.path.join(args.input_dir, "Annotations_Json/")
    args.task_name = "BM_ANPR_c27"
    args.label_list =  ['car','truck','bus','motorcyclist']

    xml_2_platform_json(args)

    # america_C28
    # args.input_dir = "/yuanhuan/data/image/RM_C28_detection/america_new/"
    # args.jpg_dir = os.path.join(args.input_dir, "JPEGImages/")
    # args.xml_dir = os.path.join(args.input_dir, "Annotations/")
    # args.platform_json_dir = os.path.join(args.input_dir, "Annotations_Json/")
    # args.task_name = "america_C28"
    # args.label_list =  ['car','truck','bus']

    # xml_2_platform_json(args)