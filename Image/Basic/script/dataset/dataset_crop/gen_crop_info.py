import argparse
import cv2
import numpy as np
import os
import pickle
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *


def getImageVar(image):
    # 利用 拉普拉斯算子 进行边缘检测，利用方差反应图像清晰度
    # 如果图片具有较高方差，那么它就有较广的频响范围，代表着正常，聚焦准确的图片
    # 如果图片具有有较小方差，那么它就有较窄的频响范围，意味着图片中的边缘数量很少，图片越模糊，其边缘就越少
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_clarity = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return img_clarity


def getImageGrayscale(image):
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_grayscale = img2gray.max()
    return img_grayscale 


def gen_info(args):
    jpg_list = get_sub_filepaths_suffix(args.jpg_dir, '.jpg')

    res_dict = {}
    for idx in tqdm(range(len(jpg_list))):
        jpg_name = os.path.basename(jpg_list[idx])
        jpg_path = jpg_list[idx]

        res_dict[jpg_name] = dict()
        res_dict[jpg_name]['name'] = jpg_name

        image = cv2.imread(jpg_path)

        # label
        img_label = os.path.basename(os.path.dirname(jpg_list[idx]))
        res_dict[jpg_name]['label'] = img_label

        # weight
        img_weight = image.shape[1]
        res_dict[jpg_name]['weight'] = img_weight

        # height
        img_height = image.shape[0]
        res_dict[jpg_name]['height'] = img_height

        # clarity
        img_clarity = getImageVar(image)
        res_dict[jpg_name]['clarity'] = img_clarity

        # grayscale
        img_grayscale = getImageGrayscale(image)
        res_dict[jpg_name]['grayscale'] = img_grayscale

        print("path: {}, label: {}, weight: {}, height: {}, clarity: {:.2f}, grayscale: {}".format(jpg_name, img_label, img_weight, img_height, img_clarity, img_grayscale))
                
    # save res
    dump_path = os.path.join(args.jpg_dir, 'res.pkl')
    f = open(dump_path, 'wb')
    pickle.dump(res_dict, f)
    f.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.input_dir = "/mnt/huanyuan2/data/image/LicensePlate/Mexico/"
    # args.jpg_dir =  args.input_dir + "Crop/"

    args.input_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan/"
    args.jpg_dir =  args.input_dir + "Crop/"

    # args.input_dir = "/mnt/huanyuan2/data/image/RM_ADAS_AllInOne/allinone_lincense_plate/" 
    # args.jpg_dir =  args.input_dir + "Crop/"

    gen_info(args)
