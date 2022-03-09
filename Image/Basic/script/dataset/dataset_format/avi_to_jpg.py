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
    create_folder(args.output_img_dir)

    cap = cv2.VideoCapture(args.avi_path) 
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
            output_img_path = os.path.join(args.output_img_dir, os.path.basename(args.avi_path).replace('.avi', '_{}.jpg'.format(frame_idx)))
            cv2.imwrite(output_img_path, img)

        frame_idx +=1
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.avi_path = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频/三道防线/0000000000000000-211125-115028-120014-00000G360540.avi"
    args.output_img_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试样本/SDFX_H2/"
    args.frame_strp = 50

    avi_to_jpg(args)
