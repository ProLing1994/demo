import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Demo.license_plate_capture_vehicle_scene.demo.RMAI_API import *
from Image.Demo.license_plate_capture_vehicle_scene.utils.draw_tools import draw_bbox_tracker, draw_bbox_info, draw_bbox_state, draw_capture_line


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
            tracker_bboxes, bbox_info_list, bbox_state_map, capture_line, capture_dict, capture_result = capture_api.run(img, frame_idx)

            if args.write_result_per_frame_bool or args.write_result_video_bool:
                # draw bbox
                # img = draw_bbox_tracker(img, tracker_bboxes)
                # img = draw_bbox_info(img, bbox_info_list, mode='ltrb')
                img = draw_bbox_info(img, bbox_info_list, capture_dict=capture_dict, mode='ltrb')
                img = draw_bbox_state(img, bbox_state_map)
                img = draw_capture_line(img, capture_line, mode='ltrb')

            # 是否保存每一帧结果
            if args.write_result_per_frame_bool:
                output_img_path = os.path.join(args.output_video_dir, video_list[idx].replace(args.suffix, ''), video_list[idx].replace(args.suffix, '_{}.jpg'.format(frame_idx)))
                create_folder(os.path.dirname(output_img_path))
                cv2.imwrite(output_img_path, img)

            # 是否保存视频结果
            if args.write_result_video_bool:
                output_video.write(img)

            # 是否保存抓拍结果
            if args.write_capture_crop_bool:
                # crop capture result
                for idy in range(len(capture_result)):
                    capture_result_idy = capture_result[idy]
                    id_idy = capture_result_idy['id']
                    plate_ocr_idy = capture_result_idy['plate_ocr']

                    for idz in range(len(capture_result_idy['img_bbox_info'])):
                        img_bbox_info_idz = capture_result_idy['img_bbox_info'][idz]
                        img_idz = img_bbox_info_idz['img']
                        bbox_info_idz = img_bbox_info_idz['bbox_info']
                        bbox_loc = [bbox_info['loc'] for bbox_info in bbox_info_idz if bbox_info['id'] == id_idy][0]
                        bbox_crop = img_idz[max( 0, bbox_loc[1] ): min( int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), bbox_loc[3] ), max( 0, bbox_loc[0] ): min( int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), bbox_loc[2] )]

                        # 保存捕获结果
                        output_capture_path = os.path.join(args.output_video_dir, 'capture', video_list[idx].replace(args.suffix, ''), '{}_{}_{}.jpg'.format(frame_idx, plate_ocr_idy, idz))
                        create_folder(os.path.dirname(output_capture_path))
                        cv2.imwrite(output_capture_path, bbox_crop)

            # 是否保存日志
            if args.write_csv_bool:

                for idy in range(len(capture_result)):
                    csv_dict = {}

                    csv_dict['name'] = video_list[idx].replace(args.suffix, '')
                    csv_dict['frame_id'] = frame_idx
                    capture_result_idy = capture_result[idy]
                    csv_dict['id'] = capture_result_idy['id']
                    csv_dict['plate'] = capture_result_idy['plate_ocr']
                    csv_dict['plate_state'] = capture_result_idy['plate_state']
                    csv_dict['plate_times'] = len(capture_result_idy['img_bbox_info'])

                    csv_list.append(csv_dict)

            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))

    # 是否保存日志
    if args.write_csv_bool:
        csv_pd = pd.DataFrame(csv_list, columns=['name', 'frame_id', 'id', 'plate', 'plate_state', 'plate_times'])
        csv_pd.to_csv(os.path.join(args.output_video_dir, 'capture.csv'), index=False, encoding="utf_8_sig")


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # zg, demo
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_HCZP/test_video/avi视频/5M_6mm_0609_白_前/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/5M_6mm_0609_白_前/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_HCZP/test_video/avi视频/5M_6mm_0609_白_后/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/5M_6mm_0609_白_后/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_HCZP/test_video/avi视频/5M_12mm_0702_白_前/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/5M_12mm_0702_白_前/" 
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_HCZP/test_video/avi视频/5M_12mm_0707_白_前/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/5M_12mm_0707_白_前/" 
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_HCZP/test_video/avi视频/5M_12mm_0713_白_前/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/5M_12mm_0713_白_前/"
    args.video_dir = "/mnt/huanyuan2/data/image/ZG_HCZP/test_video/avi视频/5M_12mm_0723_白_前/"
    args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/5M_12mm_0723_白_前/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_HCZP/test_video/avi视频/test/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/test/"
    args.suffix = '.avi'

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