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
    create_folder(args.output_video_dir)

    # video init 
    video_list = np.array(os.listdir(args.video_dir))
    video_list = video_list[[video.endswith(args.suffix) for video in video_list]]
    video_list.sort()

    for idx in tqdm(range(len(video_list))):
        video_path = os.path.join(args.video_dir, video_list[idx])

        cap = cv2.VideoCapture(video_path) 
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
                # output_img_path = os.path.join(args.output_video_dir, video_list[idx].replace(args.suffix, ''), video_list[idx].replace(args.suffix, '_{:0>5d}.jpg'.format(frame_idx)))
                output_img_path = os.path.join(args.output_video_dir, video_list[idx].replace(args.suffix, '_{:0>5d}.jpg'.format(frame_idx)))
                create_folder(os.path.dirname(output_img_path))
                cv2.imwrite(output_img_path, img)

            frame_idx +=1
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C28_safeisland/原始素材/BM_C28/2M_安全岛_0510/avi/"
    # args.output_video_dir = "/mnt/huanyuan2/data/image/RM_C28_safeisland/original/BM/jpg/JPEGImages/2M_安全岛_0510/"
    # # args.suffix = '.mp4'
    # args.suffix = '.avi'
    # # args.frame_strp = 125
    # args.frame_strp = 25
    # # args.frame_strp = 10
    # # args.frame_strp = 1

    args.video_dir = "/mnt/huanyuan2/data/image/Calibrate/Chessboard/BM1448/原始数据/棋盘格数据/2023-05-06/avi/"
    args.output_video_dir = "/mnt/huanyuan2/data/image/Calibrate/Chessboard/BM1448/原始数据/棋盘格数据/2023-05-06/jpg/"
    # args.suffix = '.mp4'
    args.suffix = '.avi'
    args.frame_strp = 25
    # args.frame_strp = 10
    # args.frame_strp = 1

    avi_to_jpg(args)
