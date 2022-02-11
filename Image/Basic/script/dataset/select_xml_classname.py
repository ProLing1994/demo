import argparse
import numpy as np
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET


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
            classname = str(object.find('name').text)
            
            for select_idx in range(0, len(args.select_name_list)):
                if args.select_name_list[select_idx] == classname:
                    object.find('name').text = args.set_name_list[select_idx]
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

    # # args.input_dir = "/yuanhuan/data/image/LicensePlate/China/"
    # # args.input_dir = "/yuanhuan/data/image/LicensePlate/China_6mm/"
    # # args.input_dir = "/yuanhuan/data/image/LicensePlate/Europe/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/Mexico/"
    # args.select_name_list = ["license_plate"]
    # args.set_name_list = ["license_plate"]
    # args.finnal_name_list = ["license_plate", "neg"]

    args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone/"
    args.select_name_list = ["car", "bus", "truck"]
    args.set_name_list = ["car", "bus", "truck"]
    args.finnal_name_list = ["car", "bus", "truck", "neg"]
        
    args.jpg_dir =  args.input_dir + "JPEGImages/"
    args.xml_dir =  args.input_dir + "XML/"
    args.output_xml_dir =  args.input_dir + "Annotations_CarBusTruckLicenseplate/"

    select_classname(args)