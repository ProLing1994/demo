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

        bool_have_pos = False       # 检测是否包含正样本标注
        bool_have_one_neg = False   # 检测是否包含一个负样本（没有正样本时生效，保证存在一个标签）
        
        # 标签检测和标签转换
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

            # 检测是否在挑选名单中
            for select_idx in range(0, len(args.select_name_list)):
                if args.select_name_list[select_idx] == classname:
                    
                    object.find('name').text = args.set_name_list[select_idx]

                    # 检测是否为车牌
                    if classname in args.plate_list:
                        # 检测是否为小车牌
                        plate_height = bndbox[3] - bndbox[1]
                        if plate_height < args.plate_height_threshold:
                            object.find('name').text = args.plate_ignore_name
                        
                        # 检测是否落在 roi ignore 区域
                        roi_ignore_bool = check_ignore_roi(bndbox, args.roi_ignore_plate_bbox)
                        if roi_ignore_bool:
                            object.find('name').text = args.roi_ignore_plate_name
                        
                    break

            if object.find('name').text in args.set_name_list:
                bool_have_pos = True
        
        # 删除无用标签
        for object in root.findall('object'):
            classname = str(object.find('name').text)

            if (classname not in args.set_name_list):
                # 如果存在正样本，删除无用标签
                if bool_have_pos:
                    root.remove(object)
                # 如果没有正样本时，生成唯一一个负样本标签（用于训练过程中的负样本）
                else:
                    print("have special_class neg file:", jpg_list[idx], "class_name is: ", classname)
                    if not bool_have_one_neg:
                        object.find('name').text = "neg"
                        bndbox = object.find('bndbox')
                        bndbox.find('xmin').text = "1900"
                        bndbox.find('ymin').text = "0"
                        bndbox.find('xmax').text = "1919"
                        bndbox.find('ymax').text = "19"
                        bool_have_one_neg = True
                    # 如果存在负样本，则不再生成标签
                    else:
                        root.remove(object)
            
        # 检测标签
        for object in root.findall('object'):
            classname = str(object.find('name').text)
            if not (classname in args.finnal_name_list):
                print(classname + "---->label is error---->" + classname)
        tree.write(output_xml_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # ######################################
    # # Annotations_CarBusTruckLicenseplate
    # # 方案一：利用 cross data training，生成 Annotations_CarBusTruckLicenseplate
    # # 正样本：清晰车牌，负样本：模糊车牌
    # # 注：忽略小于10个像素的数据
    # ######################################
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_5M/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/sandaofangxian/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/shenzhentiaoqiao/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu_night_diguangzhao/"
    # args.select_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate"]
    # args.set_name_list = ["car", "bus", "truck", "license_plate", "fuzzy_license_plate", "license_plate_ignore", "roi_ignore_plate"]
    # args.finnal_name_list = ["car", "bus", "truck", "license_plate", "fuzzy_license_plate", "license_plate_ignore", "roi_ignore_plate", "neg"]

    # # 判断大小车牌
    # args.plate_list = ['plate', "fuzzy_plate"]
    # args.plate_height_threshold = 10
    # args.plate_ignore_name = "license_plate_ignore"

    # # 标注数据添加了叠加信息，判断是否落入 roi ignore 区域
    # # args.roi_ignore_plate_bbox = [[570, 51, 1165, 97], [1761, 47, 1920, 101], [57, 983, 387, 1049]]
    # args.roi_ignore_plate_bbox = []
    # args.roi_ignore_plate_name = "roi_ignore_plate"

    # args.jpg_dir =  args.input_dir + "JPEGImages/"
    # args.xml_dir =  args.input_dir + "XML/"
    # args.output_xml_dir =  args.input_dir + "Annotations_CarBusTruckLicenseplate/"

    ######################################
    # 测试集：
    ######################################
    args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/"
    args.select_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate"]
    args.set_name_list = ["car", "bus", "truck", "license_plate", "fuzzy_license_plate", "license_plate_ignore", "roi_ignore_plate"]
    args.finnal_name_list = ["car", "bus", "truck", "license_plate", "fuzzy_license_plate", "license_plate_ignore", "roi_ignore_plate", "neg"]

    # 判断大小车牌
    args.plate_list = ['plate', "fuzzy_plate"]
    args.plate_height_threshold = 10
    args.plate_ignore_name = "license_plate_ignore"

    # 标注数据添加了叠加信息，判断是否落入 roi ignore 区域
    args.roi_ignore_plate_bbox = []
    args.roi_ignore_plate_name = "roi_ignore_plate"

    args.jpg_dir =  args.input_dir + "AHHBAS_kakou2_night/"
    args.xml_dir =  args.input_dir + "AHHBAS_kakou2_night_XML/"
    args.output_xml_dir =  args.input_dir + "AHHBAS_kakou2_night_Annotations_CarBusTruckLicenseplate"

    select_classname(args)

    # ######################################
    # # Annotations_CarBusTruckLicenseplate_w_fuzzy
    # # 方案二：只要是车牌都检测出来，通过 finetune 的方式训练
    # # 正样本：清晰车牌 & 模糊车牌
    # # 注：忽略小于10个像素的数据
    # ######################################
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_5M/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/sandaofangxian/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/shenzhentiaoqiao/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu_night_diguangzhao/"
    # args.select_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate"]
    # args.set_name_list = ["car", "bus", "truck", "license_plate", "license_plate", "license_plate_ignore", "roi_ignore_plate"]
    # args.finnal_name_list = ["car", "bus", "truck", "license_plate", "license_plate", "license_plate_ignore", "roi_ignore_plate", "neg"]

    # # 判断大小车牌
    # args.plate_list = ['plate', "fuzzy_plate"]
    # args.plate_height_threshold = 10
    # args.plate_ignore_name = "license_plate_ignore"

    # # 标注数据添加了叠加信息，判断是否落入 roi ignore 区域
    # # args.roi_ignore_plate_bbox = [[570, 51, 1165, 97], [1761, 47, 1920, 101], [57, 983, 387, 1049]]
    # args.roi_ignore_plate_bbox = []
    # args.roi_ignore_plate_name = "roi_ignore_plate"

    # args.jpg_dir =  args.input_dir + "JPEGImages/"
    # args.xml_dir =  args.input_dir + "XML/"
    # args.output_xml_dir =  args.input_dir + "Annotations_CarBusTruckLicenseplate_w_fuzzy/"

    ######################################
    # 测试集：
    ######################################
    args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/"
    args.select_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate"]
    args.set_name_list = ["car", "bus", "truck", "license_plate", "license_plate", "license_plate_ignore", "roi_ignore_plate"]
    args.finnal_name_list = ["car", "bus", "truck", "license_plate", "license_plate", "license_plate_ignore", "roi_ignore_plate", "neg"]

    # 判断大小车牌
    args.plate_list = ['plate', "fuzzy_plate"]
    args.plate_height_threshold = 10
    args.plate_ignore_name = "license_plate_ignore"

    # 标注数据添加了叠加信息，判断是否落入 roi ignore 区域
    args.roi_ignore_plate_bbox = []
    args.roi_ignore_plate_name = "roi_ignore_plate"

    args.jpg_dir =  args.input_dir + "AHHBAS_kakou2_night/"
    args.xml_dir =  args.input_dir + "AHHBAS_kakou2_night_XML/"
    args.output_xml_dir =  args.input_dir + "AHHBAS_kakou2_night_Annotations_CarBusTruckLicenseplate_w_fuzzy/"

    select_classname(args)

    # ######################################
    # # 测试方案一：高度大于阈值（24）车牌
    # # 正样本：清晰车牌 & 模糊车牌
    # # 该测试方案存在的问题：标签不统一，高度阈值 24 看似是一个定值，但标注过程中存在人为偏差
    # # 该方案：测试 高度大于阈值（24）且 清晰 & 模糊 车牌的召回率
    # ######################################
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_5M/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/sandaofangxian/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/shenzhentiaoqiao/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu_night_diguangzhao/"
    # args.select_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate"]
    # args.set_name_list = ["car", "bus", "truck", "license_plate", "license_plate", "license_plate_ignore", "roi_ignore_plate"]
    # args.finnal_name_list = ["car", "bus", "truck", "license_plate", "license_plate", "license_plate_ignore", "roi_ignore_plate", "neg"]

    # # 判断大小车牌
    # args.plate_list = ['plate', "fuzzy_plate"]
    # args.plate_height_threshold = 24
    # args.plate_ignore_name = "license_plate_ignore"

    # # 标注数据添加了叠加信息，判断是否落入 roi ignore 区域
    # # args.roi_ignore_plate_bbox = [[570, 51, 1165, 97], [1761, 47, 1920, 101], [57, 983, 387, 1049]]
    # args.roi_ignore_plate_bbox = []
    # args.roi_ignore_plate_name = "roi_ignore_plate"

    # args.jpg_dir =  args.input_dir + "JPEGImages/"
    # args.xml_dir =  args.input_dir + "XML/"
    # args.output_xml_dir =  args.input_dir + "Annotations_CarBusTruckLicenseplate_w_fuzzy_w_height/"

    ######################################
    # 测试集：
    ######################################
    args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/"
    args.select_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate"]
    args.set_name_list = ["car", "bus", "truck", "license_plate", "license_plate", "license_plate_ignore", "roi_ignore_plate"]
    args.finnal_name_list = ["car", "bus", "truck", "license_plate", "license_plate", "license_plate_ignore", "roi_ignore_plate", "neg"]

    # 判断大小车牌
    args.plate_list = ['plate', "fuzzy_plate"]
    args.plate_height_threshold = 24
    args.plate_ignore_name = "license_plate_ignore"

    # 标注数据添加了叠加信息，判断是否落入 roi ignore 区域
    args.roi_ignore_plate_bbox = []
    args.roi_ignore_plate_name = "roi_ignore_plate"

    args.jpg_dir =  args.input_dir + "AHHBAS_kakou2_night/"
    args.xml_dir =  args.input_dir + "AHHBAS_kakou2_night_XML/"
    args.output_xml_dir =  args.input_dir + "AHHBAS_kakou2_night_Annotations_CarBusTruckLicenseplate_w_fuzzy_w_height"

    select_classname(args)

    # #####################################
    # # 测试方案二：高度大于阈值（24）车牌
    # # 正样本：清晰车牌
    # # 该方案：测试 高度大于阈值（24）且 清晰 车牌的召回率
    # #####################################
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_5M/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/sandaofangxian/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/shenzhentiaoqiao/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu_night_diguangzhao/"
    # args.select_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate"]
    # args.set_name_list = ["car", "bus", "truck", "license_plate", "fuzzy_license_plate", "license_plate_ignore", "roi_ignore_plate"]
    # args.finnal_name_list = ["car", "bus", "truck", "license_plate", "fuzzy_license_plate", "license_plate_ignore", "roi_ignore_plate", "neg"]

    # # 判断大小车牌
    # args.plate_list = ['plate']
    # args.plate_height_threshold = 24
    # args.plate_ignore_name = "license_plate_ignore"

    # # 标注数据添加了叠加信息，判断是否落入 roi ignore 区域
    # # args.roi_ignore_plate_bbox = [[570, 51, 1165, 97], [1761, 47, 1920, 101], [57, 983, 387, 1049]]
    # args.roi_ignore_plate_bbox = []
    # args.roi_ignore_plate_name = "roi_ignore_plate"

    # args.jpg_dir =  args.input_dir + "JPEGImages/"
    # args.xml_dir =  args.input_dir + "XML/"
    # args.output_xml_dir =  args.input_dir + "Annotations_CarBusTruckLicenseplate_w_height/"

    ######################################
    # 测试集：
    ######################################
    args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/"
    args.select_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate"]
    args.set_name_list = ["car", "bus", "truck", "license_plate", "fuzzy_license_plate", "license_plate_ignore", "roi_ignore_plate"]
    args.finnal_name_list = ["car", "bus", "truck", "license_plate", "fuzzy_license_plate", "license_plate_ignore", "roi_ignore_plate", "neg"]

    # 判断大小车牌
    args.plate_list = ['plate']
    args.plate_height_threshold = 24
    args.plate_ignore_name = "license_plate_ignore"

    # 标注数据添加了叠加信息，判断是否落入 roi ignore 区域
    args.roi_ignore_plate_bbox = []
    args.roi_ignore_plate_name = "roi_ignore_plate"

    args.jpg_dir =  args.input_dir + "AHHBAS_kakou2_night/"
    args.xml_dir =  args.input_dir + "AHHBAS_kakou2_night_XML/"
    args.output_xml_dir =  args.input_dir + "AHHBAS_kakou2_night_Annotations_CarBusTruckLicenseplate_w_height"

    select_classname(args)