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
from Image.Demo.bus_occupying_capture.demo.RMAI_API_stabel import CaptureApi
from Image.Demo.bus_occupying_capture.utils.draw_tools import DrawApi 
from Image.Basic.script.xml.xml_write import write_xml


def inference_video(args):
    # mkdir 
    create_folder(args.output_video_dir)

    # capture api
    capture_api = CaptureApi(args.demo_type, args.country_type)

    # draw api
    draw_api = DrawApi(args.demo_type, args.country_type)

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
            tracker_bboxes, bbox_info_list, bbox_state_container, capture_line_points, capture_occupying_container, capture_occupying_res_container = capture_api.run(img, frame_idx)

            if args.write_result_per_frame_bool or args.write_result_video_bool:
                # draw bbox
                # draw_img = draw_api.draw_bbox_tracker(copy.deepcopy(img), tracker_bboxes)
                draw_img = draw_api.draw_bbox_info(copy.deepcopy(img), bbox_info_list, capture_container=capture_occupying_container, capture_res_container=capture_occupying_res_container, mode='ltrb')
                # draw_img = draw_api.draw_bbox_state(draw_img, bbox_state_container)
                draw_img = draw_api.draw_capture_line(draw_img, capture_line_points, mode='ltrb')

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
                for _, capture_res_idy in capture_occupying_res_container.items():

                    if capture_res_idy['capture_occupying']['draw_bool']:
                        continue

                    track_id = capture_res_idy['track_id']
                    num = capture_res_idy['capture_occupying']['cap_lpr_num']
                    color = capture_res_idy['capture_occupying']['cap_lpr_color']
                    capture_res_idy['capture_occupying']['draw_bool'] = True

                    for idz in range(len(capture_res_idy['capture_occupying']['cap_img'])):
                        cap_img = capture_res_idy['capture_occupying']['cap_img'][idz]
                        output_capture_path = os.path.join(args.output_video_dir, 'capture', video_list[idx].replace(args.suffix, ''), '{}_{}_{}_{}.jpg'.format(frame_idx, num, color,  idz))
                        create_folder(os.path.dirname(output_capture_path))
                        cv2.imwrite(output_capture_path, cap_img)

                        cap_plate_roi = capture_res_idy['capture_occupying']['cap_plate_roi'][idz]
                        if len(cap_plate_roi):
                            bbox_crop = cap_img[max( 0, int(cap_plate_roi[1]) ): min( int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap_plate_roi[3]) ), max( 0, int(cap_plate_roi[0]) ): min( int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_plate_roi[2]) )]
                            output_capture_path = os.path.join(args.output_video_dir, 'capture', video_list[idx].replace(args.suffix, ''), '{}_{}_{}_{}_roi.jpg'.format(frame_idx, num, color,  idz))
                            cv2.imwrite(output_capture_path, bbox_crop)

            
            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    #######################################
    # license plate
    #######################################
    # Chn, license plate demo
    args.demo_type = "lpr"
    args.country_type = "china"
    args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/RoadCap_CN_A16/20230804/avi/200W_8mm/"
    args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/RoadCap_CN_A16/20230804/200W_8mm_cap_occupying/"

    args.suffix = '.avi'
    # args.suffix = '.mp4'

    # 是否保存视频结果
    args.write_result_video_bool = True
    # 是否保存每一帧结果
    # args.write_result_per_frame_bool = True
    args.write_result_per_frame_bool = False
    # 是否保存抓拍结果
    args.write_capture_crop_bool = True

    inference_video(args)


if __name__ == '__main__':
    main()