import argparse
from tkinter import E
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *


def show_hist_clarrity(args):
    # load pickle
    dump_path = os.path.join(args.clarity_dir, 'res.pkl')
    f = open(dump_path, 'rb')
    img_dict = pickle.load(f)
    f.close()

    # init
    plate_clarity = list()
    fuzzy_plate_clarity = list()
    plate_weight = list()
    fuzzy_plate_weight = list()
    plate_height = list()
    fuzzy_plate_height = list()

    # mkdir 
    if not os.path.exists( args.height_plate_small_dir ):
        os.makedirs( args.height_plate_small_dir )
    if not os.path.exists( args.error_plate_dir ):
        os.makedirs( args.error_plate_dir )
    if not os.path.exists( args.clarity_plate_small_dir ):
        os.makedirs( args.clarity_plate_small_dir )
    if not os.path.exists( args.clarity_plate_big_dir ):
        os.makedirs( args.clarity_plate_big_dir )
    if not os.path.exists( args.clarity_plate_1000_10000_dir ):
        os.makedirs( args.clarity_plate_1000_10000_dir )
    if not os.path.exists( args.clarity_plate_10000_25000_dir ):
        os.makedirs( args.clarity_plate_10000_25000_dir )

    for key in tqdm(img_dict.keys()):
        
        img_name = img_dict[key]['name']
        img_label = img_dict[key]['label']
        img_clarity = img_dict[key]['clarity']
        img_weight = img_dict[key]['weight']
        img_height = img_dict[key]['height']

        if img_label == 'plate':
            # 过滤异常值
            if img_weight < 200:
                plate_clarity.append(img_clarity)
                plate_weight.append(img_weight)
                plate_height.append(img_height)
        elif img_label == 'fuzzy_plate':
            if img_weight < 200:
                fuzzy_plate_clarity.append(img_clarity)
                fuzzy_plate_weight.append(img_weight)
                fuzzy_plate_height.append(img_height)
        else:
            raise NotImplemented
        
        # img_height < 14，像素小于 14，残缺车牌，均为 fuzzy_plate
        if img_height < 14:
            pass
            # shutil.copy(os.path.join(args.jpg_dir, img_name), os.path.join(args.height_plate_small_dir, img_name))
            # shutil.move(os.path.join(args.jpg_dir, img_name), os.path.join(args.height_plate_small_dir, img_name))
        # img_weight > 200，像素大于 200，异常车牌，标注错误标签
        if img_weight > 200:
            pass 
            # shutil.copy(os.path.join(args.jpg_dir, img_name), os.path.join(args.error_plate_dir, img_name))
            # shutil.move(os.path.join(args.jpg_dir, img_name), os.path.join(args.error_plate_dir, img_name))
        # img_clarity < 1500，clarity 小于 1000，认为是模糊车牌
        if img_clarity < 1000:
            pass
            # shutil.copy(os.path.join(args.jpg_dir, img_name), os.path.join(args.clarity_plate_small_dir, img_name))
            # shutil.move(os.path.join(args.jpg_dir, img_name), os.path.join(args.clarity_plate_small_dir, img_name))
        # img_clarity> 25000，clarity 大于 25000，认为是清晰车牌
        if img_clarity > 25000:
            pass
            # shutil.copy(os.path.join(args.jpg_dir, img_name), os.path.join(args.clarity_plate_big_dir, img_name))
            # shutil.move(os.path.join(args.jpg_dir, img_name), os.path.join(args.clarity_plate_big_dir, img_name))
        if img_clarity >= 1000 and img_clarity < 10000:
            pass
            # shutil.copy(os.path.join(args.jpg_dir, img_name), os.path.join(args.clarity_plate_1000_10000_dir, img_name))
            # shutil.move(os.path.join(args.jpg_dir, img_name), os.path.join(args.clarity_plate_1000_10000_dir, img_name))
        if img_clarity > 10000 and img_clarity < 25000:
            pass
            # shutil.copy(os.path.join(args.jpg_dir, img_name), os.path.join(args.clarity_plate_10000_25000_dir, img_name))
            # shutil.move(os.path.join(args.jpg_dir, img_name), os.path.join(args.clarity_plate_10000_25000_dir, img_name))
    
    # clarity
    plate_np = np.array(plate_clarity)
    fuzzy_plate_np = np.array(fuzzy_plate_clarity)
    plt.hist(x = plate_np, bins = 100, color = 'red', alpha=0.5, edgecolor = 'black')
    plt.hist(x = fuzzy_plate_np, bins = 100, color = 'green', alpha=0.5, edgecolor = 'black')

    plt.show()
    plt.close()

    # weight
    plate_np = np.array(plate_weight)
    fuzzy_plate_np = np.array(fuzzy_plate_weight)
    plt.hist(x = plate_np, bins = 100, color = 'red', alpha=0.5, edgecolor = 'black')
    plt.hist(x = fuzzy_plate_np, bins = 100, color = 'green', alpha=0.5, edgecolor = 'black')

    plt.show()
    plt.close()

    # height
    plate_np = np.array(plate_height)
    fuzzy_plate_np = np.array(fuzzy_plate_height)
    plt.hist(x = plate_np, bins = 100, color = 'red', alpha=0.5, edgecolor = 'black')
    plt.hist(x = fuzzy_plate_np, bins = 100, color = 'green', alpha=0.5, edgecolor = 'black')
    print(plate_np.min(), fuzzy_plate_np.min())

    plt.show()
    plt.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/"
    args.jpg_dir =  args.input_dir + "LicensePlate_crop/"
    args.clarity_dir = args.input_dir + "LicensePlate_crop_clarity/"
    args.height_plate_small_dir = args.input_dir + "LicensePlate_crop_height_plate_small/"
    args.error_plate_dir = args.input_dir + "LicensePlate_crop_error/"
    args.clarity_plate_small_dir = args.input_dir + "LicensePlate_crop_clarity_plate_small/"
    args.clarity_plate_big_dir = args.input_dir + "LicensePlate_crop_clarity_plate_big/"
    args.clarity_plate_1000_10000_dir = args.input_dir + "LicensePlate_crop_clarity_plate_1000_10000/"
    args.clarity_plate_10000_25000_dir = args.input_dir + "LicensePlate_crop_clarity_plate_10000_25000/"

    show_hist_clarrity(args)