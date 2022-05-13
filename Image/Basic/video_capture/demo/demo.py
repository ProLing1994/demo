import argparse
import cv2
import multiprocessing
from moviepy import editor
import numpy as np
import os
import pandas as pd
import sys 
import torch
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, "E:\\project\\demo")
from Image.Basic.utils.folder_tools import *
from Image.Basic.video_capture.demo.VideoCapture_API import *


# def video_capture_csv(args, video_name):
def video_capture_csv(in_params):
    args = in_params[0]
    video_name = in_params[1]

    output_csv_path = os.path.join(args.output_video_dir, video_name.replace(args.suffix, '.csv'))
    if os.path.exists(output_csv_path):
        return

    # video capture api
    video_capture_api = VideoCaptureApi()

    # pd init
    video_capture_list = []
    video_capture_dict = {}
    video_capture_dict['id'] = 0
    video_capture_dict['start_s'] = 0
    video_capture_dict['end_s'] = 0
    video_capture_dict['attri'] = 0
    video_capture_dict['plate_color'] = 0

    video_path = os.path.join(args.video_dir, video_name)

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
        
        if frame_idx % args.step_frame != 0:
            frame_idx += 1
            continue

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

        tqdm.write("{}: {}".format(video_name, str(frame_idx)))

    # out csv
    csv_data_pd = pd.DataFrame(video_capture_list)
    csv_data_pd.to_csv(output_csv_path, index=False, encoding="utf_8_sig")
    return 


def vidio_capture_merge(args, video_name):

    video_path = os.path.join(args.video_dir, video_name)
    capture_csv_path = os.path.join(args.output_video_dir, video_name.replace(args.suffix, '.csv'))

    if not os.path.exists(capture_csv_path):
        return 

    # pd init
    video_capture_list = []
    video_capture_dict = {}
    video_capture_dict['start_s'] = 0
    video_capture_dict['end_s'] = 0

    # capture_pd
    try:
        capture_pd = pd.read_csv(capture_csv_path, encoding='utf_8_sig')
    except pd.errors.EmptyDataError as error:
        print(error)
        return 

    for idx, row in capture_pd.iterrows():
        start_s_idx = row['start_s']
        end_s_idx = row['end_s']
        
        if len(video_capture_list):
            if video_capture_list[-1]['end_s'] + 2 * args.time_shift_s >= start_s_idx:
                video_capture_list[-1]['start_s'] = min( video_capture_list[-1]['start_s'], start_s_idx )
                video_capture_list[-1]['end_s'] = max( video_capture_list[-1]['end_s'], end_s_idx )
            else:
                video_capture_dict = {}
                video_capture_dict['start_s'] = row['start_s']
                video_capture_dict['end_s'] = row['end_s']
                video_capture_list.append(video_capture_dict)
        else:
            video_capture_dict = {}
            video_capture_dict['start_s'] = row['start_s']
            video_capture_dict['end_s'] = row['end_s']
            video_capture_list.append(video_capture_dict)

    # out csv
    output_csv_path = os.path.join(args.output_video_dir, video_name.replace(args.suffix, '_merge.csv'))
    csv_data_pd = pd.DataFrame(video_capture_list)
    csv_data_pd.to_csv(output_csv_path, index=False, encoding="utf_8_sig")
    return


# def vidio_capture_crop(args, video_name):
def vidio_capture_crop(in_params):
    args = in_params[0]
    video_name = in_params[1]

    video_path = os.path.join(args.video_dir, video_name)
    capture_csv_path = os.path.join(args.output_video_dir, video_name.replace(args.suffix, '.csv'))

    if not os.path.exists(capture_csv_path):
        return 

    # capture_pd
    try:
        capture_pd = pd.read_csv(capture_csv_path, encoding='utf_8_sig')
    except pd.errors.EmptyDataError as error:
        print(error)
        return 

    for idx, row in capture_pd.iterrows():
        # VideoFileClip 
        try:
            video_clip = editor.VideoFileClip(video_path)
        except:
            continue

        # info 
        attri_idx = row['attri']
        plate_color_idx = row['plate_color']
        start_s_idx = max(0, float(row['start_s']) - args.time_shift_s ) 
        end_s_idx = min(video_clip.end, float(row['end_s']) + args.time_shift_s)

        # 挑选车牌颜色
        if args.select_plate_color != None:
            if plate_color_idx != args.select_plate_color:
                continue
        
        # 挑选车型
        if args.select_car_attri != None:
            if attri_idx != args.select_car_attri:
                continue

        # 超出视频末尾，原因未知
        if start_s_idx >= video_clip.end:
            continue
        
        # 超长视频裁剪
        if end_s_idx - start_s_idx > args.time_threshold:
            start_s_idx = end_s_idx - args.time_threshold
        
        print(start_s_idx, end_s_idx)
        # mkdir
        output_video_path = os.path.join(args.output_video_dir, '{}_{}'.format(args.select_plate_color, args.select_car_attri), video_name.replace(args.suffix, ''), '{}_{:.2f}_{:.2f}_{}_{}.mp4'.format(video_name.replace(args.suffix, ''), start_s_idx, end_s_idx, attri_idx, plate_color_idx))
        create_folder(os.path.dirname(output_video_path))
        if os.path.exists(output_video_path):
            print("ignore path: {}".format(output_video_path))
            continue

        # crop
        try:
            clip = video_clip.subclip(start_s_idx, end_s_idx)
            clip.write_videofile(output_video_path, verbose=True)
        except:
            pass

    return 

# def vidio_capture_crop_merge(args, video_name):
def vidio_capture_crop_merge(in_params):
    args = in_params[0]
    video_name = in_params[1]

    video_path = os.path.join(args.video_dir, video_name)
    capture_csv_path = os.path.join(args.output_video_dir, video_name.replace(args.suffix, '_merge.csv'))

    if not os.path.exists(capture_csv_path):
        return 

    # capture_pd
    try:
        capture_pd = pd.read_csv(capture_csv_path, encoding='utf_8_sig')
    except pd.errors.EmptyDataError as error:
        print(error)
        return 

    for idx, row in capture_pd.iterrows():
        # VideoFileClip 
        try:
            video_clip = editor.VideoFileClip(video_path)
        except:
            continue
    
        # info 
        start_s_idx = max(0, float(row['start_s']) - args.time_shift_s ) 
        end_s_idx = min(video_clip.end, float(row['end_s']) + args.time_shift_s)

        # 超出视频末尾，原因未知
        if start_s_idx >= video_clip.end:
            continue

        print(start_s_idx, end_s_idx)
        # mkdir
        output_video_path = os.path.join(args.output_video_dir, 'merge', video_name.replace(args.suffix, ''), '{}_{:.2f}_{:.2f}.mp4'.format(video_name.replace(args.suffix, ''), start_s_idx, end_s_idx))
        create_folder(os.path.dirname(output_video_path))
        if os.path.exists(output_video_path):
            print("ignore path: {}".format(output_video_path))
            continue
        
        # crop
        try:
            clip = video_clip.subclip(start_s_idx, end_s_idx)
            clip.write_videofile(output_video_path, verbose=True)
        except:
            pass

    return 


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.video_dir = "X:\\temp\\卡口1\\2022-04-10\\avi"
    args.output_video_dir = "X:\\temp\\卡口1\\2022-04-10\\avi_video_capture"
    args.suffix = '.avi'

    # args.video_dir = "/mnt/huanyuan/temp/天桥/2022-04-01/avi/"
    # args.output_video_dir = "/mnt/huanyuan/temp/天桥/2022-04-01/avi_video_capture/"
    # args.suffix = '.avi'

    args.step_frame = 3

    # mkdir 
    create_folder(args.output_video_dir)

    # video init 
    video_list = np.array(os.listdir(args.video_dir))
    video_list = video_list[[video.endswith(args.suffix) for video in video_list]]
    video_list.sort()

    in_params = []
    for idx in tqdm(range(len(video_list))):
        in_args = [args, video_list[idx]]
        in_params.append(in_args)

    # # step 1: 
    # # 车辆抓取
    # ctx = torch.multiprocessing.get_context("spawn")
    # p = ctx.Pool(1)
    # out = p.map(video_capture_csv, in_params)
    # p.close()
    # p.join()

    # # 截取视频段，前后扩展时间
    # args.time_shift_s = 3

    # # step 2: 
    # # 视频合并
    # for idx in tqdm(range(len(video_list))):
    #     vidio_capture_merge(args, video_list[idx])

    # # step 3: 
    # # 视频剪裁
    # p = multiprocessing.Pool(2)
    # out = p.map(vidio_capture_crop_merge, in_params)
    # p.close()
    # p.join()

    # 截取视频段，前后扩展时间
    args.time_shift_s = 3

    # 截取视频段，最长时间
    args.time_threshold = 20

    # step 4: 
    # 挑选颜色
    args.select_plate_color = 'yellow'

    # 挑选车型
    args.select_car_attri = None
    
    # for idx in range(len(in_params)):
    #     vidio_capture_crop(in_params[idx])

    # step 5: 
    # 视频剪裁
    p = multiprocessing.Pool(3)
    out = list(tqdm(p.map(vidio_capture_crop, in_params), total=len(in_params)))
    p.close()
    p.join()

    # # step 6: 
    # args.select_plate_color = 'green'

    # # 挑选车型
    # args.select_car_attri = None

    # # step 7: 
    # # 视频剪裁
    # p = multiprocessing.Pool(3)
    # out = list(tqdm(p.map(vidio_capture_crop, in_params), total=len(in_params)))
    # p.close()
    # p.join()


if __name__ == '__main__':
    main()