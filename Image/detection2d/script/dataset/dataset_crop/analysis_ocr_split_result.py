import cv2
import numpy as np
import os 
import argparse
import shutil
import sys

from sympy import deg

sys.path.insert(0, '/home/huanyuan/code/demo/')
from Image.recognition2d.license_plate_ocr.infer.lpr import LPR
from Image.Basic.utils.folder_tools import *


def gen_ocr_result(args):
    # init
    lpr = LPR(args.lpr_caffe_prototxt, args.lpr_caffe_model_path, args.lpr_prefix_beam_search_bool)

    for label_idx in range(len(args.select_label)):
        img_label = args.select_label[label_idx]
        image_list = os.listdir(os.path.join(args.image_dir, img_label))

        for idx in range(len(image_list)):
            image_path = os.path.join(args.image_dir, img_label, image_list[idx])
            img = cv2.imread(image_path, 0) 
            result_lstm, result_scors_list = lpr.run(img)

            if np.array(result_scors_list).mean() < args.ocr_threshold:
                output_path = os.path.join(args.image_dir, 'ocr_result_{}'.format(args.ocr_threshold), "{}_fuzzy".format(img_label), image_list[idx])
                create_folder( os.path.dirname(output_path) )
                shutil.copy(image_path, output_path)
                print (image_list[idx], result_lstm, np.array(result_scors_list).mean())
            else:
                output_path = os.path.join(args.image_dir, 'ocr_result_{}'.format(args.ocr_threshold), "{}_clear".format(img_label), image_list[idx])
                create_folder( os.path.dirname(output_path) )
                shutil.copy(image_path, output_path)

    return 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # china: lpr_lxn
    args.lpr_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr_lxn/china_softmax.prototxt"
    args.lpr_caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr_lxn/china.caffemodel"
    args.lpr_prefix_beam_search_bool = False

    # args.image_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan/Crop_itering/height_0_24/"
    args.image_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan/Crop_itering/height_24_200/"
    args.select_label = ["plate", "fuzzy_plate"]

    args.ocr_threshold = 0.8

    gen_ocr_result(args)