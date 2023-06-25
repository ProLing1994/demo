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

    args.input_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_crop/shate_20230308_old"
    args.output_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_crop/shate_20230308"

    args.img_folder = "Images"
    args.json_folder = "Json"
    args.xml_folder = "xml"

    input_json_dir = os.path.join(args.input_dir, args.json_folder)
    output_img_dir = os.path.join(args.output_dir, args.img_folder)
    output_json_dir = os.path.join(args.output_dir, args.json_folder)

    img_list = os.listdir(output_img_dir)
    for img_idx in tqdm(range(len(img_list))):
        img = img_list[img_idx]
        img_name = ("-").join(img.split("-")[:-1])
        plate_name = img.replace(".jpg", "").split('_')[-1]
        
        find_json_name = ""
        xml_list = os.listdir(input_json_dir)
        for xml_idx in range(len(xml_list)):
            xml_name = xml_list[xml_idx]
            if img_name in xml_name and plate_name in xml_name:
                find_json_name = xml_name
                break
        # print(img, find_json_name)
        if find_json_name == "":
            print()
            continue
        
        json_path = os.path.join(input_json_dir, find_json_name) 
        out_json_path = os.path.join(output_json_dir, img.replace(".jpg", ".json"))

        # print(xml_path, out_xml_path)
        create_folder(os.path.dirname(out_json_path))
        shutil.copy(json_path, out_json_path)