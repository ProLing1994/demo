import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *


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
        cv2.imwrite(output_jpg_path, img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/mnt/huanyuan/model_final/image_model/lpr_zd/ocr_image/"
    args.output_dir = "/mnt/huanyuan/model_final/image_model/lpr_zd/ocr_image_gray/"
    args.suffix = '.jpg'

    bgr_to_gray(args)
