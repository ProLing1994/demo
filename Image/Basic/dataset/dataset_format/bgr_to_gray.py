import argparse
import cv2
import math
import numpy as np
import os
from tqdm import tqdm
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
# sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def pad_ratio(img, image_shape, bool_white=False):

    img = img[: img.shape[0]//4*4, : img.shape[1]//4*4]

    h, w = img.shape[0], img.shape[1]
    imgH, imgW = image_shape

    max_wh_ratio = imgW * 1.0 / imgH
    ratio = w * 1.0 / h

    # pad
    if ratio < max_wh_ratio:
        to_imgW = int(math.ceil(h * max_wh_ratio)) // 4 * 4
        if (to_imgW - w < 32):
            to_imgW = w + 32

        if bool_white:
            pad_img = np.ones((h, to_imgW), dtype=np.uint8)
            pad_img *= 255
        else:
            pad_img = np.zeros((h, to_imgW), dtype=np.uint8)

        pad_img[:, 0:w] = img  
        img = pad_img  

    else:
        img = img

    return img


def bgr_to_gray(args):
    # mkdir 
    create_folder(args.output_dir)

    # jpg init 
    jpg_list = np.array(os.listdir(args.input_dir))
    jpg_list = jpg_list[[jpg.endswith(args.suffix) for jpg in jpg_list]]
    jpg_list.sort()

    for idx in tqdm(range(len(jpg_list))):
        jpg_path = os.path.join(args.input_dir, jpg_list[idx])
        output_jpg_path = os.path.join(args.output_dir, jpg_list[idx])

        img = cv2.imread(jpg_path, 0)
        if args.bool_pad:
            img = pad_ratio(img, args.image_shape, bool_white=args.bool_white)
        cv2.imwrite(output_jpg_path, img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/mnt/huanyuan/model_final/image_model/lpr_zd/ocr_image/"
    args.output_dir = "/mnt/huanyuan/model_final/image_model/lpr_zd/ocr_image_gray_white_ratio_64_320/"
    args.suffix = '.jpg'
    args.bool_pad = True
    args.bool_white = True
    args.image_shape = [64, 320]

    bgr_to_gray(args)
