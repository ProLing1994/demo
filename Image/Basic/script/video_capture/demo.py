import argparse
from tkinter import Image
import cv2
from moviepy import editor
import numpy as np
import os
import pandas as pd
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Basic.script.video_capture.VideoCapture_API import *


def video_capture_csv(args):
    # mkdir 
    create_folder(args.output_video_dir)

    # video capture api
    video_capture_api = VideoCaptureApi()

    # video init 
    video_list = np.array(os.listdir(args.video_dir))
    video_list = video_list[[video.endswith(args.suffix) for video in video_list]]
    video_list.sort()

    # pd init
    video_capture_list = []
    video_capture_dict = {}
    video_capture_dict['id'] = 0
    video_capture_dict['start_frame'] = 0
    video_capture_dict['end_frame'] = 0
    video_capture_dict['attri'] = 0
    video_capture_dict['plate_color'] = 0
    video_capture_dict['frame_rate'] = 0

    for idx in tqdm(range(len(video_list))):
        video_path = os.path.join(args.video_dir, video_list[idx])

        cap = cv2.VideoCapture(video_path) 
        print(int(cap.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧

        frame_idx = 0

        # video capture api 
        # 输入新视频，状态复位
        video_capture_api.clear()

        while True:
            ret, img = cap.read()

            if not ret: # if the camera over return false
                # video_writer.release()
                break
            
            # if frame_idx == 178:
            #     print()

            # video capture api 
            # capture_id_list：需要抓拍的车辆id
            # capture_info：抓拍结果
            bbox_info_list, capture_id_list, capture_info_list = video_capture_api.run(img, frame_idx)

            for idy in range(len(capture_info_list)):
                capture_info_idy = capture_info_list[idy]
                
                video_capture_dict = {}
                video_capture_dict['id'] = capture_info_idy['id']
                video_capture_dict['start_s'] = float(capture_info_idy['start_frame']) / float(cap.get(cv2.CAP_PROP_FPS))
                video_capture_dict['end_s'] = float(capture_info_idy['end_frame']) / float(cap.get(cv2.CAP_PROP_FPS))
                video_capture_dict['attri'] = capture_info_idy['attri']
                video_capture_dict['plate_color'] = capture_info_idy['plate_color']
                video_capture_list.append(video_capture_dict)

            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))

        # out csv
        output_csv_path = os.path.join(args.output_video_dir, video_list[idx].replace(args.suffix, '.csv'))
        create_folder(os.path.dirname(output_csv_path))
        csv_data_pd = pd.DataFrame(video_capture_list)
        csv_data_pd.to_csv(output_csv_path, index=False, encoding="utf_8_sig")


def vidio_capture_crop(args):

    # video init 
    video_list = np.array(os.listdir(args.video_dir))
    video_list = video_list[[video.endswith(args.suffix) for video in video_list]]
    video_list.sort()
    
    for idx in tqdm(range(len(video_list))):
        video_path = os.path.join(args.video_dir, video_list[idx])
        video_name = video_list[idx]

        capture_csv_path = os.path.join(args.output_video_dir, video_name.replace(args.suffix, '.csv'))
        capture_pd = pd.read_csv(capture_csv_path, encoding='utf_8_sig')

        for idx, row in capture_pd.iterrows():
            # VideoFileClip 
            video_clip = editor.VideoFileClip(video_path)
        
            # info 
            attri_idx = row['attri']
            plate_color_idx = row['plate_color']
            start_s_idx = max(0, float(row['start_s']) - args.time_shift_s ) 
            end_s_idx = min(video_clip.duration, float(row['end_s']) + args.time_shift_s)

            if end_s_idx - start_s_idx > args.max_time_s:
                start_s_idx = max(0, end_s_idx - args.max_time_s)

            # mkdir
            output_video_path = os.path.join(args.output_video_dir, video_name.replace(args.suffix, ''), '{}_{:.2f}_{:.2f}_{}_{}.mp4'.format(video_name.replace(args.suffix, ''), start_s_idx, end_s_idx, attri_idx, plate_color_idx))
            create_folder(os.path.dirname(output_video_path))

            # crop
            clip = video_clip.subclip(start_s_idx, end_s_idx)
            clip.write_videofile(output_video_path, verbose=True)

def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频实验/test/"
    # args.output_video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频实验/test_video_capture/"
    # args.suffix = '.avi'

    args.video_dir = "/mnt/huanyuan/temp/卡口2/2022-03-29/temp/"
    args.output_video_dir = "/mnt/huanyuan/temp/卡口2/2022-03-29/temp_video_capture/"
    args.suffix = '.avi'

    # video_capture_csv(args)

    # 截取视频段，前后扩展时间
    args.time_shift_s = 3
    # 截取视频段，最长时间
    args.max_time_s = 16.0

    vidio_capture_crop(args)


if __name__ == '__main__':
    main()