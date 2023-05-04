import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm


sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Demo.car_capture_zd_bus_safezone.demo.RMAI_API import *
from Image.Demo.car_capture_zd_bus_safezone.utils.draw_tools import draw_bbox_tracker, draw_bbox_info, draw_bbox_state, draw_capture_line


def inference_video(args):
    # mkdir 
    create_folder(args.output_video_dir)

    # capture api
    capture_api = CaptureApi()

    # video init 
    video_list = np.array(os.listdir(args.video_dir))
    video_list = video_list[[video.endswith(args.suffix) for video in video_list]]
    video_list.sort()

    if args.write_csv_bool:
        csv_list = []

    for idx in tqdm(range(len(video_list))):
        video_path = os.path.join(args.video_dir, video_list[idx])

        cap = cv2.VideoCapture(video_path) 
        print(int(cap.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧
        
        # check
        if (cap.get(cv2.CAP_PROP_FRAME_WIDTH) != capture_api.image_width or
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT) != capture_api.image_height):
            continue

        output_video_path = os.path.join(args.output_video_dir, video_list[idx].replace(args.suffix, ''), video_list[idx])
        create_folder(os.path.dirname(output_video_path))

        frame_idx = 0

        # capture api 
        # 输入新视频，状态复位
        capture_api.clear()

        # 是否保存视频结果
        if args.write_result_video_bool:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (capture_api.image_width, capture_api.image_height), True)

        while True:
            ret, img = cap.read()

            if not ret: # if the camera over return false
                break
                
            # if frame_idx == 17:
            #     print()

            # capture api
            tracker_bboxes, bbox_info_list, bbox_state_map, capture_points = capture_api.run(img, frame_idx)

            if args.write_result_per_frame_bool or args.write_result_video_bool:
                # draw bbox
                # img = draw_bbox_tracker(img, tracker_bboxes)
                img = draw_capture_line(img, capture_points)
                img = draw_bbox_info(img, bbox_info_list, capture_points, mode='ltrb')
                img = draw_bbox_state(img, bbox_state_map)

            # 是否保存每一帧结果
            if args.write_result_per_frame_bool:
                output_img_path = os.path.join(args.output_video_dir, video_list[idx].replace(args.suffix, ''), video_list[idx].replace(args.suffix, '_{}.jpg'.format(frame_idx)))
                create_folder(os.path.dirname(output_img_path))
                cv2.imwrite(output_img_path, img)

            # 是否保存视频结果
            if args.write_result_video_bool:
                output_video.write(img)

            # 是否保存日志
            if args.write_csv_bool:
                pass

            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))

    # 是否保存日志
    if args.write_csv_bool:
        pass


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # zd, safezone, demo
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C28_detection/test_video/ZD_safezone_C28/avi/moni_0329/校车靠右边车道（0米）"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_safezone_C28/moni_0329/校车靠右边车道（0米）"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C28_detection/test_video/ZD_safezone_C28/avi/moni_0329/校车靠左边车道（1米）"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_safezone_C28/moni_0329/校车靠左边车道（1米）"
    args.video_dir = "/mnt/huanyuan2/data/image/RM_C28_detection/test_video/ZD_safezone_C28/avi/moni_0329/校车倾斜"
    args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_safezone_C28/moni_0329/校车倾斜"

    args.suffix = '.avi'
    # args.suffix = '.mp4'

    # 是否保存视频结果
    args.write_result_video_bool = True
    # 是否保存每一帧结果
    # args.write_result_per_frame_bool = True
    args.write_result_per_frame_bool = False
    # 是否保存抓拍结果
    args.write_capture_crop_bool = True
    # 是否保存日志
    args.write_csv_bool = True

    inference_video(args)


if __name__ == '__main__':
    main()