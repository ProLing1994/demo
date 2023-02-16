import argparse
import numpy as np
import os
from tqdm import tqdm
import sys 
import shutil
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def find_xml_label(args):
    # mkdir 
    if not os.path.exists( args.output_dir ):
        os.makedirs( args.output_dir )

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]

    for idx in tqdm(range(len(jpg_list))):
        jpg_path = os.path.join(args.jpg_dir, jpg_list[idx])
        output_jpg_path = os.path.join(args.output_dir, jpg_list[idx])

        xml_path = os.path.join(args.xml_dir, jpg_list[idx].replace(".jpg", ".xml"))
        output_xml_path = os.path.join(args.output_dir, jpg_list[idx].replace(".jpg", ".xml"))

        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

        bool_have_pos = False

        for object in root.findall('object'):

            # name
            classname = str(object.find('name').text.lower().strip())

            if classname in args.find_label_list:
                
                bool_have_pos = True
        
        if bool_have_pos:
            print(jpg_path)
            shutil.copy(jpg_path, output_jpg_path)
            shutil.copy(xml_path, output_xml_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_MS1/"
    args.jpg_dir = args.input_dir + "JPEGImages/"  
    args.xml_dir = args.input_dir + "Annotations_CarBusTruckMotorcyclePlateMotoplate_w_fuzzy/"
    args.find_label_list = ["license_plate"]

    args.output_dir = args.input_dir + "plate_JPEGImages/"

    find_xml_label(args)
