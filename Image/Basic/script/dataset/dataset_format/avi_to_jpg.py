import argparse
import cv2
from importlib.resources import path
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *


def avi_to_jpg(args):
    # mkdir 
    create_folder(args.output_vidio_dir)

    # vidio init 
    vidio_list = np.array(os.listdir(args.vidio_dir))
    vidio_list = vidio_list[[vidio.endswith(args.suffix) for vidio in vidio_list]]
    vidio_list.sort()

    for idx in tqdm(range(len(vidio_list))):
        vidio_path = os.path.join(args.vidio_dir, vidio_list[idx])

        cap = cv2.VideoCapture(vidio_path) 
        print(int(cap.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧

        frame_idx = 0
        while True:

            ret, img = cap.read()

            if not ret: # if the camera over return false
                break         

            if frame_idx % args.frame_strp == 0:
                output_img_path = os.path.join(args.output_vidio_dir, vidio_list[idx].replace('.avi', ''), vidio_list[idx].replace('.avi', '_{}.jpg'.format(frame_idx)))
                create_folder(os.path.dirname(output_img_path))
                cv2.imwrite(output_img_path, img)

            frame_idx +=1
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.vidio_dir = "/mnt/huanyuan2/data/image/ZG_AHHBGS_detection/原始数据视频/002500041C/2022-03-18/avi/"
    args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_AHHBGS_detection/test/002500041C/2022-03-18/avi_to_jpg/"
    args.suffix = '.avi'
    args.frame_strp = 10

    avi_to_jpg(args)
