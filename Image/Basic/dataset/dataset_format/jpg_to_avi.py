import argparse
import cv2
from importlib.resources import path
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *


def jpg_to_avi(args):
    # mkdir 
    create_folder(args.output_video_dir)
    
    # file
    file_list = os.listdir(args.jpg_dir)

    for idx in tqdm(range(len(file_list))):
        file_name = file_list[idx]
        file_path = os.path.join(args.jpg_dir, file_name)

        # img
        img_list = np.array(os.listdir(file_path))
        img_list = img_list[[img.endswith(args.suffix) for img in img_list]]
        img_list.sort()

        # video
        video_path = os.path.join(args.output_video_dir, '{}.avi'.format(file_name))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(video_path, fourcc, 20.0, args.video_shape, True)

        for idy in tqdm(range(len(img_list))):
            img_name = img_list[idy]
            img_path = os.path.join(file_path, img_name)

            # img
            img = cv2.imread(img_path)

            output_video.write(img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.jpg_dir = "/mnt/huanyuan/temp/pc_demo/test_jpg/"
    args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/test_video/"
    args.suffix = '.jpg'
    args.video_shape = (2592, 1920)

    jpg_to_avi(args)