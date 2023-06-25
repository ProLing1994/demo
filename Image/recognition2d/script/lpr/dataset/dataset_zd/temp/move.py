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

    args.input_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/temp/UAE/check"
    args.output_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_old_style/uae_2022_old"

    args.img_folder = "JPEGImages"
    args.json_folder = "Json"
    args.xml_folder = "xml"

    file_list = get_sub_filepaths_suffix(args.input_dir, '.jpg')

    for file_idx in tqdm(range(len(file_list))):
        file = file_list[file_idx]

        img_name = os.path.basename(file)
        img_path = file
        out_img_path = os.path.join(args.output_dir, args.img_folder, img_name)
        # print(img_path, out_img_path)
        create_folder(os.path.dirname(out_img_path))
        # shutil.copy(img_path, out_img_path)
        shutil.move(img_path, out_img_path)

        try:
            xml_name = img_name.replace('.jpg', '.xml')
            xml_path = os.path.join(os.path.dirname(img_path), xml_name)
            out_xml_path = os.path.join(args.output_dir, args.xml_folder, xml_name)
            create_folder(os.path.dirname(out_xml_path))
            # shutil.copy(xml_path, out_xml_path)
            shutil.move(xml_path, out_xml_path)
        except:
            pass

        try:
            json_name = img_name.replace('.jpg', '.json')
            json_path = os.path.join(os.path.dirname(img_path), json_name)
            out_json_path = os.path.join(args.output_dir, args.json_folder, json_name)
            create_folder(os.path.dirname(out_json_path))
            # shutil.copy(json_path, out_json_path)
            shutil.move(json_path, out_json_path)
        except:
            pass