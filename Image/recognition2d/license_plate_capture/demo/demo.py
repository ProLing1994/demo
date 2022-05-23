import argparse
from tkinter import Image
import cv2
import numpy as np
import os
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.recognition2d.license_plate_capture.demo.RMAI_API import *
from Image.regreesion2d.plate_regreesion.utils.draw_tools import draw_bbox_info, draw_capture_line


def inference_video(args):
    # mkdir 
    create_folder(args.output_video_dir)

    # capture api
    capture_api = CaptureApi()

    # image init 
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

        while True:
            ret, img = cap.read()

            if not ret: # if the camera over return false
                break
            
            # if frame_idx == 304:
            #     print()

            # capture api
            # bbox_info_list：单检测识别结果
            # capture_line：抓拍线
            # capture_id_list：需要抓拍的车辆id
            # capture_result：抓拍结果
            bbox_info_list, capture_line, capture_id_list, capture_result = capture_api.run(img, frame_idx)

            # crop license plate
            for idy in range(len(bbox_info_list)):
                bbox_info_idy = bbox_info_list[idy]
                plate_ocr_idy = bbox_info_idy['plate_ocr']
                
                if bbox_info_idy['plate_crop_bool']:
                    # car
                    crop_img = img[bbox_info_idy['plate_loc'][1]:bbox_info_idy['plate_loc'][3], bbox_info_idy['plate_loc'][0]:bbox_info_idy['plate_loc'][2]]
                    output_crop_img_path = os.path.join(args.output_video_dir, 'plate_crop', video_list[idx].replace(args.suffix, ''), '{}_{}.jpg'.format(plate_ocr_idy, frame_idx))
                    create_folder(os.path.dirname(output_crop_img_path))
                    cv2.imwrite(output_crop_img_path, crop_img)

            # draw bbox
            img = draw_bbox_info(img, bbox_info_list, capture_id_list, mode='ltrb')
            img = draw_capture_line(img, capture_line, mode='ltrb')

            output_img_path = os.path.join(args.output_video_dir, video_list[idx].replace(args.suffix, ''), video_list[idx].replace(args.suffix, '_{}.jpg'.format(frame_idx)))
            create_folder(os.path.dirname(output_img_path))
            cv2.imwrite(output_img_path, img)

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

            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # # zg，智观加油站数据 2M
    # # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_ZHJYZ_220119/264原始视频/2M/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_ZHJYZ_220119/264原始视频/2M_big/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_ZHJYZ_220119/test/"
    # # args.output_video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_ZHJYZ_220119/pc_20220406_车牌抓拍实验/2M_叠加及抓拍结果/"
    # args.output_video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_ZHJYZ_220119/pc_20220414_车牌抓拍实验_车辆状态优化/2M_叠加及抓拍结果/"
    # args.suffix = '.avi'

    # zg，智观加油站数据 5M
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频实验/5MH_2lane/"
    # args.output_video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频实验/5MH_2lane_capture/"
    # args.suffix = '.avi'

    # # zg，安徽淮北高速 5M
    # # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/264原始视频/264原始视频/5M_12mm_卡口2_白_0327/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/264原始视频/264原始视频/5M_12mm_卡口2_白_0401/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_AHHBGS/264原始视频/5M_12mm_卡口3_晚_0407/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_AHHBGS/264原始视频/5M_12mm_卡口3_白_0407/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_AHHBGS/264原始视频/5M_12mm_卡口2_白_0406/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_AHHBGS/264原始视频/5M_12mm_卡口2_白_0408/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_AHHBGS/264原始视频/5M_12mm_卡口2_白_0420/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_AHHBGS/264原始视频/5M_12mm_卡口1_白_0427/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_AHHBGS/264原始视频/temp/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_AHHBGS/原视频结果/5M_12mm_卡口1_卡口2_0506_原视频结果/卡口1-新/2022-05-06/record-sub/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_AHHBGS/原视频结果/5M_12mm_卡口1_卡口2_0506_原视频结果/卡口2/2022-05-06/record-sub/"
    # # args.output_video_dir = "/home/huanyuan/temp/pc_车牌抓拍实验_车辆状态优化_模型迭代更新/5M_12mm_卡口1_白_0427_叠加及抓拍结果/"
    # args.output_video_dir = "/home/huanyuan/temp/pc_车牌抓拍实验_车辆状态优化_模型迭代更新_版本0430/5M_12mm_卡口1_0506_叠加及抓拍结果/"
    # args.suffix = '.avi'

    # zg，深圳天桥 5M
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_TQ/264原始视频/5M_12mm_白_0414/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_TQ/264原始视频/5M_16mm_白_0414/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_TQ/264原始视频/5M_12mm_白_0419/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_TQ/264原始视频/5M_16mm_白_0419/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_TQ/264原始视频/5M_12mm_16M_白_0424/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_TQ/264原始视频/5M_16mm_16M_白_0424/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_TQ/264原始视频/5M_16mm_16M_白_0424/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_TQ/264原始视频/5M_16mm_6M_白_0506/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_TQ/264原始视频/5M_12mm_6M_白_0506/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_TQ/264原始视频/5M_16mm_16M_白_8m_20m_60km_0518/"
    args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_TQ/264原始视频/5M_12mm_16M_白_8m_20m_60km_0518/"
    args.output_video_dir = "/home/huanyuan/temp/pc_车牌抓拍实验_车辆状态优化_模型迭代更新_版本0430/5M_12mm_16M_白_8m_20m_60km_0518_叠加及抓拍结果/"
    # args.output_video_dir = "/home/huanyuan/temp/test/"
    args.suffix = '.avi'

    # args.video_dir = "/mnt/huanyuan/test/avi/"
    # args.output_video_dir = "/home/huanyuan/temp/test/avi_capture/"
    # args.suffix = '.avi'

    inference_video(args)


if __name__ == '__main__':
    main()