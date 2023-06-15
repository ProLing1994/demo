import argparse
import cv2
import copy
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *


def avi_split(args):

    # mkdir 
    create_folder(args.output_video_dir)

    # video init 
    video_list = np.array(os.listdir(args.video_dir))
    video_list = video_list[[video.endswith(args.suffix) for video in video_list]]
    video_list.sort()

    for idx in tqdm(range(len(video_list))):
        video_name = video_list[idx]
        video_path = os.path.join(args.video_dir, video_list[idx])

        cap = cv2.VideoCapture(video_path) 
        print(int(cap.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧

        frame_idx = 0
        output_video = None
        output_video_name_idx = 0
        
        while True:

            ret, img = cap.read()

            if not ret: # if the camera over return false
                break         

            if frame_idx % (args.max_second * int(cap.get(cv2.CAP_PROP_FPS))) == 0:
                if output_video != None:
                    output_video.release() 

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_video_path = os.path.join(args.output_video_dir, video_name.replace(args.suffix, '_{:0>5d}.avi'.format(output_video_name_idx)))
                output_video = cv2.VideoWriter(output_video_path, fourcc, float(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), True)
                output_video_name_idx += 1
            else:
                output_video.write(img)
            frame_idx +=1


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_人脸测试数据_前方反馈_20230606/C40W_2688_1520/"
    args.output_video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_人脸测试数据_前方反馈_20230606/C40W_2688_1520_split/"
    args.suffix = '.mp4'

    args.max_second = 60 * 2

    avi_split(args)