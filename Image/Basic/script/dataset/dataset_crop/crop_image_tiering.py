import argparse
import os
import pickle
import shutil
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *


def image_itering(args):
    # load pickle
    dump_path = os.path.join(args.jpg_dir, 'res.pkl')
    f = open(dump_path, 'rb')
    img_dict = pickle.load(f)
    f.close()

    for key in tqdm(img_dict.keys()):
        
        img_name = img_dict[key]['name']
        img_label = img_dict[key]['label']
        img_clarity = img_dict[key]['clarity']
        img_weight = img_dict[key]['weight']
        img_height = img_dict[key]['height']

        if not img_label in args.select_label:
            continue
        
        for height_idx in range(len(args.height_list)):
            min_height = args.height_list[height_idx][0]
            max_height = args.height_list[height_idx][1]

            if img_height > min_height and img_height <= max_height:
                
                input_img_path = os.path.join(args.jpg_dir, img_label, img_name)
                output_img_path = os.path.join(args.itering_dir, "height_{}_{}".format(min_height, max_height), img_label, img_name)
                create_folder( os.path.dirname(output_img_path) )

                shutil.copy(input_img_path, output_img_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.input_dir = "/mnt/huanyuan2/data/image/RM_ADAS_AllInOne/allinone_lincense_plate/"
    # args.jpg_dir =  args.input_dir + "Crop/"
    # args.itering_dir = args.input_dir + "Crop_itering/"
    
    # args.select_label = ['plate', 'fuzzy_plate']
    # args.height_list = [[0, 15], [15, 20], [20, 35], [35, 200]]

    args.input_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan/"
    args.jpg_dir =  args.input_dir + "Crop/"
    args.itering_dir = args.input_dir + "Crop_itering/"
    
    args.select_label = ['plate', 'fuzzy_plate']
    args.height_list = [[0, 24], [24, 200]]
    image_itering(args)