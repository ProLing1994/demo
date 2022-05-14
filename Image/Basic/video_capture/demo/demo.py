import argparse
import cv2
import multiprocessing
from moviepy import editor
import numpy as np
import os
import pandas as pd
import sys 
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, 'E:\\project\\demo')
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
    video_capture_api = VideoCaptureApi(args.model_prototxt, args.model_path, args.GPU)

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
        # bbox_info_list：单检测识别结果
        # capture_id_list：需要抓拍的车辆id
        # capture_result：抓拍结果
        bbox_info_list, capture_id_list, capture_result = video_capture_api.run(img, frame_idx)

        for idy in range(len(capture_result)):
            capture_info_idy = capture_result[idy]
            
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

    # VideoFileClip 
    try:
        video_clip = editor.VideoFileClip(video_path)
    except:
        return

    for idx, row in capture_pd.iterrows():
    
        # info 
        attri_idx = row['attri']
        plate_color_idx = row['plate_color']
        start_s_idx = max(0, float(row['start_s']) - args.time_shift_s ) 
        end_s_idx = min(video_clip.end, float(row['end_s']) + args.time_shift_s)

        # 挑选车型或颜色
        if len(args.select_plate_color) != 0 and len(args.select_car_attri) != 0:
            if plate_color_idx not in args.select_plate_color and attri_idx not in args.select_car_attri:
                continue
        # 挑选车牌颜色
        elif len(args.select_plate_color) != 0:
            if plate_color_idx not in args.select_plate_color:
                continue
        # 挑选车型
        elif len(args.select_car_attri) != 0:
            if attri_idx not in args.select_car_attri:
                continue

        # 超出视频末尾，原因未知
        if start_s_idx >= video_clip.end:
            continue

        # 超长视频裁剪
        if end_s_idx - start_s_idx > args.time_threshold:
            start_s_idx = end_s_idx - args.time_threshold

        print(start_s_idx, end_s_idx)
        # mkdir
        output_video_path = os.path.join(args.output_video_dir, '_'.join(args.select_plate_color + args.select_car_attri), video_name.replace(args.suffix, ''), '{}_{:.2f}_{:.2f}_{}_{}.mp4'.format(video_name.replace(args.suffix, ''), start_s_idx, end_s_idx, attri_idx, plate_color_idx))
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

    # VideoFileClip 
    try:
        video_clip = editor.VideoFileClip(video_path)
    except:
        return

    for idx, row in capture_pd.iterrows():

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
    parser.add_argument('--video_dir', type=str, default="E:\\test\\avi") 
    parser.add_argument('--output_video_dir', type=str, default="E:\\test\\avi_video_capture/") 
    parser.add_argument('--suffix', type=str, default='.avi') 
    parser.add_argument('--steps', type=str, default='1,2,3') 
    parser.add_argument('--model_prototxt', type=str, default=None) 
    parser.add_argument('--model_path', type=str, default="E:\\project\\model\\image\\ssd_rfb\\SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-04-25-18.xml") 
    parser.add_argument('--GPU', action='store_true', default=False) 

    # args = parser.parse_args()
    args, unparsed = parser.parse_known_args() 
    
    # option
    args.step_frame = 2

    # 截取视频段，前后扩展时间
    args.time_shift_s = 3

    # 截取视频段，最长时间
    args.time_threshold = 20

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

    # steps
    step_list = str(args.steps).split(',')

    if '1' in step_list:
        # step 1: 
        # 车辆抓取
        # import torch
        # ctx = torch.multiprocessing.get_context("spawn")
        # p = ctx.Pool(2)
        # out = p.map(video_capture_csv, in_params)
        # p.close()
        # p.join()

        p = multiprocessing.Pool(2)
        out = p.map(video_capture_csv, in_params)
        p.close()
        p.join()

        # video_capture_csv(in_params[0])

    if '2' in step_list:
        # step 2: 
        # 视频合并
        for idx in tqdm(range(len(video_list))):
            vidio_capture_merge(args, video_list[idx])

        # 视频剪裁
        p = multiprocessing.Pool(2)
        out = p.map(vidio_capture_crop_merge, in_params)
        p.close()
        p.join()

    if '3' in step_list:
        # step 3: 
        # 挑选颜色：黄色
        args.select_plate_color = ['yellow']

        # 挑选车型
        args.select_car_attri = ['truck', 'bus']
            
        # 视频剪裁
        p = multiprocessing.Pool(2)
        out = list(tqdm(p.map(vidio_capture_crop, in_params), total=len(in_params)))
        p.close()
        p.join()

    if '4' in step_list:
        # step 4: 
        # 挑选颜色：黄色
        args.select_plate_color = [ ]

        # 挑选车型
        args.select_car_attri = ['truck', 'bus']
            
        # 视频剪裁
        p = multiprocessing.Pool(2)
        out = list(tqdm(p.map(vidio_capture_crop, in_params), total=len(in_params)))
        p.close()
        p.join()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()