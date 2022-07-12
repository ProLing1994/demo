import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys
from pyrsistent import v 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Demo.zebra_crossing_detection.demo.RMAI_API import *
from Image.Demo.zebra_crossing_detection.utils.draw_tools import draw_bbox_info


def inference_video(args):
    # mkdir 
    create_folder(args.output_video_dir)

    # capture api
    capture_api = CaptureApi()
    
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

        if args.write_csv_bool:
            csv_list = []

        while True:
            ret, img = cap.read()

            if not ret: # if the camera over return false
                break

            # if frame_idx == 17:
            #     print()

            # capture api
            tracker_bboxes, bbox_info_list = capture_api.run(img, frame_idx)

            # 是否保存每一帧结果
            if args.write_result_per_frame_bool:
                # draw bbox
                img = draw_bbox_info(img, bbox_info_list, mode='ltrb')

                output_img_path = os.path.join(args.output_video_dir, video_list[idx].replace(args.suffix, ''), video_list[idx].replace(args.suffix, '_{}.jpg'.format(frame_idx)))
                create_folder(os.path.dirname(output_img_path))
                cv2.imwrite(output_img_path, img)

            if args.write_result_video_bool:
                output_video.write(img)

            if args.write_csv_bool:
                for idy in range(len(bbox_info_list)):
                    
                    csv_dict = {}
                    bbox_info_idx = bbox_info_list[idy]

                    csv_dict['frame_id'] = frame_idx
                    csv_dict['label'] = bbox_info_idx['label']
                    csv_dict['id'] = bbox_info_idx['id']
                    csv_dict['loc'] = bbox_info_idx['loc']               
                    csv_list.append(csv_dict)
                
            frame_idx += 1
            
            tqdm.write("{}: {}".format(video_path, str(frame_idx)))

        if args.write_csv_bool:
            out_csv_path = os.path.join(args.output_video_dir, video_list[idx].replace(args.suffix, ''), video_list[idx].replace(args.suffix, '.csv'))
            csv_data_pd = pd.DataFrame(csv_list)
            csv_data_pd.to_csv(out_csv_path, index=False, encoding="utf_8_sig")

def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # zg, demo
    args.video_dir = "/mnt/huanyuan2/data/image/ZG_BMX_detection/banmaxian_test_video/ZG_RongHeng/avi视频/test/"
    args.output_video_dir = "/home/huanyuan/temp/pc_demo/test/"
    args.suffix = '.avi'

    # 是否保存视频结果
    args.write_result_video_bool = True
    # 是否保存每一帧结果
    args.write_result_per_frame_bool = True
    # 是否保存日志
    args.write_csv_bool = True

    inference_video(args)

if __name__ == '__main__':
    main()