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


def show_hist_info(args):
    # load pickle
    dump_path = os.path.join(args.jpg_dir, 'res.pkl')
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

    for key in tqdm(img_dict.keys()):
        
        img_name = img_dict[key]['name']
        img_label = img_dict[key]['label']
        img_clarity = img_dict[key]['clarity']
        img_weight = img_dict[key]['weight']
        img_height = img_dict[key]['height']

        if img_label == 'plate':
            plate_clarity.append(img_clarity)
            plate_weight.append(img_weight)
            plate_height.append(img_height)
        elif img_label == 'fuzzy_plate':
            fuzzy_plate_clarity.append(img_clarity)
            fuzzy_plate_weight.append(img_weight)
            fuzzy_plate_height.append(img_height)
        else:
            pass
    
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

    plt.show()
    plt.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.input_dir = "/mnt/huanyuan2/data/image/RM_ADAS_AllInOne/allinone_lincense_plate/"
    # args.jpg_dir =  args.input_dir + "Crop/"

    args.input_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan/"
    args.jpg_dir =  args.input_dir + "Crop/"

    show_hist_info(args)