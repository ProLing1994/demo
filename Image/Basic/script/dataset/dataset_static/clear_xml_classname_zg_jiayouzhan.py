import argparse
import numpy as np
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET


def check_ignore_roi(in_box, roi_ignore_list):
    roi_ignore_bool = False

    for idx in range(len(roi_ignore_list)):
        roi_ignore_bbox = roi_ignore_list[idx]

        if in_box[0] > roi_ignore_bbox[0] and in_box[0] < roi_ignore_bbox[2] and in_box[1] > roi_ignore_bbox[1] and in_box[1] < roi_ignore_bbox[3]:
            roi_ignore_bool = True
        
        if roi_ignore_bool:
            break

    return roi_ignore_bool


def check_img_exist(in_img_name, path_list):
    img_exist_bool = False

    for idx in range(len(path_list)):
        img_path = os.path.join(path_list[idx], in_img_name)
        if os.path.exists(img_path):
            img_exist_bool = True

    return img_exist_bool


def select_classname(args):
    # mkdir 
    if not os.path.exists( args.output_xml_dir ):
        os.makedirs( args.output_xml_dir )

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]

    for idx in tqdm(range(len(jpg_list))):
        xml_path = os.path.join(args.xml_dir, jpg_list[idx].replace(".jpg", ".xml"))
        output_xml_path = os.path.join(args.output_xml_dir, jpg_list[idx].replace(".jpg", ".xml"))

        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点
        
        # 标签检测和标签转换
        idy = 0
        for object in root.findall('object'):
            # name
            classname = object.find('name').text.lower().strip()

            # bbox
            bbox = object.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                bndbox.append(cur_pt)

            if classname in args.select_name_list:
                img_name = os.path.join(jpg_list[idx].replace(".jpg", "_{}.jpg".format(idy)))
                idy += 1

            # 检测是否在挑选名单中
            for select_idx in range(0, len(args.select_name_list)):
                if args.select_name_list[select_idx] == classname:
                    
                    object.find('name').text = args.set_name_list[select_idx]

                    # 检测是否为车牌
                    if classname in ['plate', "fuzzy_plate"]:
                      
                        # 检测是否落在 roi ignore 区域
                        roi_ignore_bool = check_ignore_roi(bndbox, args.roi_ignore_plate_bbox)
                        if roi_ignore_bool:
                            object.find('name').text = args.roi_ignore_plate_name
                    
                    # 检测是否为模糊车牌
                    if classname in ["fuzzy_plate"]:
                        
                        # 检测是否位于 fuzzy_plate_clear 文件夹中
                        img_exist_bool = check_img_exist(img_name, args.fuzzy_plate_clear_path_list)
                        if img_exist_bool:
                            object.find('name').text = args.change_plate_name
                    
            
        # 检测标签
        for object in root.findall('object'):
            classname = str(object.find('name').text)
            if not (classname in args.finnal_name_list):
                print(classname + "---->label is error---->" + classname)
        tree.write(output_xml_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # 清洗数据标注
    args.input_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan/"
    args.select_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate"]
    args.set_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate", "roi_ignore_plate"]
    args.finnal_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate", "roi_ignore_plate"]

    # 标注数据添加了叠加信息，判断是否落入 roi ignore 区域
    args.roi_ignore_plate_bbox = [[570, 51, 1165, 97], [1761, 47, 1920, 101], [57, 983, 387, 1049]]
    args.roi_ignore_plate_name = "roi_ignore_plate"

    # 模糊车牌，ocr 阈值判断为清晰车牌的车牌，更改标注为清晰车牌
    args.fuzzy_plate_clear_path_list = [
                                        "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan/Crop_itering/height_0_24/ocr_result_0.8/fuzzy_plate_clear/",
                                        "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan/Crop_itering/height_24_200/ocr_result_0.8/fuzzy_plate_clear/",
                                        ]
    args.change_plate_name = "plate"

    args.jpg_dir =  args.input_dir + "JPEGImages/"
    args.xml_dir =  args.input_dir + "XML/"
    args.output_xml_dir =  args.input_dir + "XML_Clean/"

    select_classname(args)