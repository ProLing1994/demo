import argparse
import cv2
import io
import json
import os
import pandas as pd
import shutil
import sys
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.input_data_list = ["check_crop_0804_0809", "check_crop_0810_0811", "check_crop_0828_0831", "check_crop_0901_0903", "check_crop_0904_0905", "check_crop_1024_1029", "check_crop_1024_1029_1080p", "check_crop_1115_1116", "check_crop_1115_1116_1080p"]
    # args.input_data_list = [ "check_crop_0", "check_crop_1", "check_crop_2", "check_crop_3" ]
    # args.input_data_list = [ "check_crop_4", "check_crop_5", "check_crop_6", "check_crop_7", "check_crop_8" ]
    args.input_data_list = [ "check_crop_city_AJMAN_0", "check_crop_city_FUJAIRAH_0", "check_crop_city_RAK_0", "check_crop_city_SHARJAH_0", "check_crop_city_UMMALQAIWAIN_0", "check_crop_color_green_0", "check_crop_color_yellow_0" ]

    args.input_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/temp/UAE_crop/"
    args.output_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_crop/"

    args.img_folder = "Images"
    args.xml_folder = "xml"

    for idx in tqdm(range(len(args.input_data_list))):

        folder_name = args.input_data_list[idx]
        input_xml_dir = os.path.join(args.input_dir, folder_name, args.xml_folder)

        # output_folder_name = folder_name.replace("check_crop_", "uae_2022")
        output_folder_name = folder_name.replace("check_crop", "uae_2022")
        output_dir = os.path.join(args.output_dir, output_folder_name)
        output_img_dir = os.path.join(output_dir, args.img_folder)
        output_xml_dir = os.path.join(output_dir, args.xml_folder)

        img_list = os.listdir(output_img_dir)
        for img_idx in tqdm(range(len(img_list))):
            img = img_list[img_idx]
            img_name = ("-").join(img.split("-")[:-1])
            plate_name = img.replace(".jpg", "").split('_')[-1]
            
            find_xml_name = ""
            xml_list = os.listdir(input_xml_dir)
            for xml_idx in range(len(xml_list)):
                xml_name = xml_list[xml_idx]
                if img_name in xml_name and plate_name in xml_name:
                    find_xml_name = xml_name
                    break
            # print(img, find_xml_name)
            if find_xml_name == "":
                print()
                continue
            
            xml_path = os.path.join(input_xml_dir, find_xml_name) 
            out_xml_path = os.path.join(output_xml_dir, img.replace(".jpg", ".xml"))

            # print(xml_path, out_xml_path)
            create_folder(os.path.dirname(out_xml_path))
            shutil.copy(xml_path, out_xml_path)