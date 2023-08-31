import argparse
import numpy as np
import os
from PIL import Image
import sys 
import shutil
import torch
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *

def find_xml(args):
    
    # mkdir 
    create_folder(args.output_xml_dir)
    
    # jpg list
    jpg_list = np.array(os.listdir(args.input_img_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list.sort()

    for idx in tqdm(range(len(args.find_subfolder_list))): 
        
        find_subfolder = args.find_subfolder_list[idx]
        find_dir = os.path.join(args.find_dir, find_subfolder, 'image')

        # jpg list
        find_jpg_list = os.listdir(find_dir)

        for idy in range(len(find_jpg_list)): 

            find_jpg = find_jpg_list[idy]
            
            if find_jpg in jpg_list:

                finx_xml_path = os.path.join(args.find_dir, find_subfolder, 'xml', find_jpg.replace('.jpg', '.xml'))
                to_xml_path = os.path.join(args.output_xml_dir, find_jpg.replace('.jpg', '.xml'))
                shutil.copy(finx_xml_path, to_xml_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="wallet") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/Taxi_remnant/original_select") 
    parser.add_argument('--find_dir', type=str, default="/yuanhuan/data/image/Taxi_remnant/original/shenzhen") 
    args = parser.parse_args()

    args.find_subfolder_list = ['BYDe6_middle_20230720', 'BYDe6_side_20230720', 'Camry_middle_20230719', 'Camry_middle_20230720', 'Camry_side_20230719', 'Fox_20230809', 'Havel_20230810', 'Lexus_20230810', 'MKZ_20230807', 'MKZ_middle_20230721', 'MKZ_side_20230721', 'Pickup_middle_20230721', 'SUV_20230809']
    args.input_dir = os.path.join(args.input_dir, args.date_name)
    args.output_dir = args.input_dir

    print("prepare dataset.")
    print("date_name: {}".format(args.date_name))
    print("input_dir: {}".format(args.input_dir))
    print("output_dir: {}".format(args.output_dir))

    args.input_img_dir = os.path.join(args.input_dir, 'image')
    args.output_xml_dir = os.path.join(args.output_dir, 'xml')

    find_xml(args)