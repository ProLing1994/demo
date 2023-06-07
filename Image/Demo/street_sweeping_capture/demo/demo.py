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
from Image.Demo.street_sweeping_capture.demo.RMAI_API_stabel import CaptureApi
from Image.Demo.street_sweeping_capture.utils.draw_tools import DrawApi
from Image.Basic.script.xml.xml_write import write_xml


def inference_video(args):
    # mkdir 
    create_folder(args.output_video_dir)

    # capture api
    capture_api = CaptureApi(args.demo_type, args.country_type)

    # draw api
    draw_api = DrawApi(args.demo_type,)

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
            output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (capture_api.options.image_width, capture_api.options.image_height), True)

        while True:
            ret, img = cap.read()

            if not ret: # if the camera over return false
                break

            # if frame_idx == 19:
            #     print()

            # capture api
            tracker_bboxes, bbox_info_list, bbox_state_container, capture_line_up_down, capture_line_left_right, capture_container, capture_res_container = capture_api.run(img, frame_idx)

            if args.write_result_per_frame_bool or args.write_result_video_bool:
                # draw bbox
                # draw_img = draw_api.draw_bbox_tracker(copy.deepcopy(img), tracker_bboxes)
                draw_img = draw_api.draw_bbox_info(copy.deepcopy(img), bbox_info_list, capture_container=capture_container, capture_res_container=capture_res_container, mode='ltrb')
                # draw_img = draw_api.draw_bbox_state(draw_img, bbox_state_container)
                # draw_img = draw_api.draw_capture_line(draw_img, capture_line_up_down, capture_line_left_right, mode='ltrb')

            # 是否保存每一帧结果
            if args.write_result_per_frame_bool:
                output_img_path = os.path.join(args.output_video_dir, video_list[idx].replace(args.suffix, ''), video_list[idx].replace(args.suffix, '_{:0>5d}.jpg'.format(frame_idx)))
                create_folder(os.path.dirname(output_img_path))
                cv2.imwrite(output_img_path, draw_img)

            # 是否保存视频结果
            if args.write_result_video_bool:
                output_video.write(draw_img)

            # 是否保存抓拍结果
            if args.write_capture_crop_bool:
                # crop capture result
                for _, capture_res_idy in capture_res_container.items():

                    if capture_res_idy['capture']['draw_bool']:
                        continue

                    track_id = capture_res_idy['track_id']
                    flage = capture_res_idy['capture']['flage']
                    capture_res_idy['capture']['draw_bool'] = True

                    for idz in range(len(capture_res_idy['capture']['img_bbox_info_list'])):
                        img_bbox_info = capture_res_idy['capture']['img_bbox_info_list'][idz]
                        img_crop = img_bbox_info['img']
                        bbox_info = img_bbox_info['bbox_info']

                        if args.demo_type == "lpr":
                            num = capture_res_idy['plate_info']['num']
                            color = capture_res_idy['plate_info']['color']
                            bbox_loc = [bbox_info['car_info']['roi'] for bbox_info in bbox_info if bbox_info['track_id'] == track_id][0]
                            output_capture_path = os.path.join(args.output_video_dir, 'capture', video_list[idx].replace(args.suffix, ''), '{}_{}_{}_{}_{}.jpg'.format(frame_idx, num, color, flage, idz))
                        elif args.demo_type == "face":
                            landmark_degree = bbox_info[0]['face_info']['landmark_degree']
                            bbox_loc = [bbox_info['face_info']['roi'] for bbox_info in bbox_info if bbox_info['track_id'] == track_id][0]
                            output_capture_path = os.path.join(args.output_video_dir, 'capture', video_list[idx].replace(args.suffix, ''), '{}_{}_{}_{:.2f}.jpg'.format(frame_idx, flage, idz, landmark_degree))

                        bbox_crop = img_crop[max( 0, int(bbox_loc[1]) ): min( int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(bbox_loc[3]) ), max( 0, int(bbox_loc[0]) ): min( int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(bbox_loc[2]) )]

                        # 保存捕获结果
                        create_folder(os.path.dirname(output_capture_path))
                        cv2.imwrite(output_capture_path, bbox_crop)
            
            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # # zg, license plate demo
    # args.demo_type = "lpr"
    # args.country_type = "china"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/5M_230416/avi/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/lpr_avi/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/lpr_avi_paddle_0601/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/5M_12mm_0723_白_前/avi/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_12mm_0723_白_前/lpr_avi/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/Distance_detection/Distance_detection_plate/avi/200W/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/Distance_detection_plate/200W/"
    # args.video_dir = "/mnt/huanyuan2/data/image/Distance_detection/Distance_detection_plate/avi/720p/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/Distance_detection_plate/720p/"
    
    # zg, face demo
    args.demo_type = "face"
    args.country_type = "none"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/5M_230416/avi/"
    args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/5M_230416/test/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/face_avi/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/face_avi_landmark/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/face_avi_landmark_center_offset/"
    args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/face_avi_landmark_sigmoid_center_offset_qa_0604/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/face_avi_landmark_degree/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/test/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_2M_全_多方向_人脸_1024_1115/avi/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_ZD_DUBAI_C27/5M_2M_全_多方向_人脸_1024_1115/face_avi/"

    args.suffix = '.avi'
    # args.suffix = '.mp4'

    # 是否保存视频结果
    args.write_result_video_bool = True
    # 是否保存每一帧结果
    args.write_result_per_frame_bool = True
    # args.write_result_per_frame_bool = False
    # 是否保存抓拍结果
    args.write_capture_crop_bool = True

    inference_video(args)


if __name__ == '__main__':
    main()