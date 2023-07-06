import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Demo.license_plate_capture_zd_police.demo.RMAI_API import *
from Image.Demo.license_plate_capture_zd_police.utils.draw_tools import draw_bbox_info, draw_bbox_state, draw_capture_line, draw_bbox_info_result_jpg


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

        if cap.get(cv2.CAP_PROP_FRAME_WIDTH) != capture_api.image_width or cap.get(cv2.CAP_PROP_FRAME_HEIGHT) != capture_api.image_height:
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

        # 是否保存原始图像
        if args.write_ori_jpg_bool:
            txt_detect_list = []

        while True:
            ret, ori_img = cap.read()
            if not ret: # if the camera over return false
                break

            img = ori_img.copy()

            # if frame_idx == 17:
            #     print()

            # capture api
            bboxes, tracker_bboxes, bbox_info_list, bbox_state_map, capture_line_up_down, capture_line_left_right, capture_list, capture_res_list = capture_api.run(img, frame_idx)

            if args.write_result_per_frame_bool or args.write_result_video_bool:
                img = draw_bbox_info(img, bbox_info_list, capture_list=capture_list, mode='ltrb')
                # img = draw_bbox_state(img, bbox_state_map)
                # img = draw_capture_line(img, capture_line_up_down, capture_line_left_right, mode='ltrb')

            # 是否保存每一帧结果
            if args.write_result_per_frame_bool:
                output_img_path = os.path.join(args.output_video_dir, video_list[idx].replace(args.suffix, ''), video_list[idx].replace(args.suffix, '_{}.jpg'.format(frame_idx)))
                create_folder(os.path.dirname(output_img_path))
                cv2.imwrite(output_img_path, img)

            # 是否保存预测结果图像
            if args.write_result_jpg_bool:
                result_jpg_img = np.zeros((100, 1000, 3))
                result_jpg_img = draw_bbox_info_result_jpg(result_jpg_img, bbox_info_list)

                output_img_path = os.path.join(args.output_video_dir, video_list[idx].replace(args.suffix, ''), video_list[idx].replace(args.suffix, '_{}_res.jpg'.format(frame_idx)))
                cv2.imwrite(output_img_path, result_jpg_img)

            # 是否保存视频结果
            if args.write_result_video_bool:
                output_video.write(img)
            
            # 是否保存抓拍结果
            if args.write_capture_crop_bool:
                # crop capture result
                for idy, capture_res_idy in capture_res_list.items():

                    if capture_res_idy['draw_bool']:
                        continue

                    id = capture_res_idy['id']
                    country = capture_res_idy['country']
                    city = capture_res_idy['city']
                    car_type = capture_res_idy['car_type']
                    color = capture_res_idy['color']
                    kind = capture_res_idy['kind']
                    num = capture_res_idy['num']
                    column = capture_res_idy['column']
                    img_bbox_info_list = capture_res_idy['img_bbox_info_list']
                    capture_res_idy['draw_bool'] = True

                    for idz in range(len(img_bbox_info_list)):
                        img_bbox_info = img_bbox_info_list[idz]
                        img_crop = img_bbox_info['img']
                        bbox_info = img_bbox_info['bbox_info']

                        bbox_loc = [bbox_info['loc'] for bbox_info in bbox_info if bbox_info['id'] == id][0]
                        bbox_crop = img_crop[max( 0, bbox_loc[1] ): min( int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), bbox_loc[3] ), max( 0, bbox_loc[0] ): min( int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), bbox_loc[2] )]

                        # 保存捕获结果
                        output_capture_path = os.path.join(args.output_video_dir, 'capture', video_list[idx].replace(args.suffix, ''), '{}_{}_{}_{}_{}_{}_{}_{}_{}.jpg'.format(frame_idx, country, city, car_type, color, column, kind, num, idz))
                        create_folder(os.path.dirname(output_capture_path))
                        cv2.imwrite(output_capture_path, bbox_crop)
                    
                    # 是否保存日志
                    if args.write_csv_bool:
                        csv_dict = {}

                        csv_dict['name'] = video_list[idx].replace(args.suffix, '')
                        csv_dict['frame_id'] = frame_idx
                        csv_dict['id'] = capture_res_idy['id']
                        csv_dict['country'] = capture_res_idy['country']
                        csv_dict['city'] = capture_res_idy['city']
                        csv_dict['kind'] = capture_res_idy['kind']
                        csv_dict['num'] = capture_res_idy['num']
                        csv_dict['column'] = capture_res_idy['column']
                        csv_dict['flage'] = capture_res_idy['flage']

                        csv_list.append(csv_dict)
                    
            # 是否保存原始图像
            if args.write_ori_jpg_bool:

                txt_detect_str = ""
                txt_detect_str = txt_detect_str + video_list[idx].replace(args.suffix, '_{}.jpg'.format(frame_idx)) + ";"

                for key in bboxes.keys():
                    for idb in range(len(bboxes[key])):
                        txt_detect_str = txt_detect_str + str(key) + ","
                        txt_detect_str = txt_detect_str + str(bboxes[key][idb][0]) + ","
                        txt_detect_str = txt_detect_str + str(bboxes[key][idb][1]) + ","
                        txt_detect_str = txt_detect_str + str(bboxes[key][idb][2]) + ","
                        txt_detect_str = txt_detect_str + str(bboxes[key][idb][3]) + ";"

                txt_detect_list.append(txt_detect_str)

                # 保存原图
                output_jpg_path = os.path.join(args.output_video_dir, 'jpg', video_list[idx].replace(args.suffix, ''), video_list[idx].replace(args.suffix, '_{}.jpg'.format(frame_idx)))
                create_folder(os.path.dirname(output_jpg_path))
                cv2.imwrite(output_jpg_path, ori_img)
            
            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))

        # 是否保存原始图像
        if args.write_ori_jpg_bool:
            output_txt_path = os.path.join(args.output_video_dir, 'jpg', video_list[idx].replace(args.suffix, ''), video_list[idx].replace(args.suffix, '.txt'))
            with open(output_txt_path, "w") as f:
                for idw in range(len(txt_detect_list)):
                    f.writelines(txt_detect_list[idw]) 
                    f.writelines("\n") 

    # 是否保存日志
    if args.write_csv_bool:
        csv_pd = pd.DataFrame(csv_list, columns=['name', 'frame_id', 'id', 'country', 'city', 'kind', 'num', 'column', 'flage'])
        csv_pd.to_csv(os.path.join(args.output_video_dir, 'capture.csv'), index=False, encoding="utf_8_sig")


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # zd, ZD_DUBAI
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_白天_侧向_0615/截取视频/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_白天_侧向_0615/截取视频/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//test//"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_白天_侧向_0615/截取视频/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_夜晚_侧向_0615/截取视频/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_夜晚_侧向_0615/截取视频/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_白天_后向_0615/截取视频/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_白天_后向_0615/截取视频/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_夜晚_后向_0615/00000G000170/截取视频/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_夜晚_后向_0615/00000G000170/截取视频/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_夜晚_后向_0615/00000G000171/截取视频/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_夜晚_后向_0615/00000G000171/截取视频/"
    args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_白天_侧向_0615/avi/"
    args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_白天_侧向_0615/20230629/"
    
    # zd, ZD_SCH_BUS
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/POLICE_ZD_C27/test/avi/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/SchBus_ZD_AD_C27_C28/POLICE_ZD_C27/test/"

    # # zd, car type
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//车牌车型/PUBLIC/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/PUBLIC/test/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//车牌车型/POLICE/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/POLICE/test/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//车牌车型/TAXI/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/TAXI/test/"

    # # # zd, city
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//车牌城市/AD/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/AD/test/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//车牌城市/AJMAN/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/AJMAN/test/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//车牌城市/SHJ/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/SHJ/test/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//车牌城市/RAK/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/RAK/test/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//车牌城市/DXB/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/DXB/test/"

    # zd, 误报
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_全_多方向_0905/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_全_多方向_0905/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_全_多方向_0904/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_全_多方向_0904/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_全_多方向_0903/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_全_多方向_0903/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_2M_全_多方向_误报_1024/2022-10-24_14/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_2M_全_多方向_误报_1024/2022-10-24_14/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_2M_全_多方向_误报_1024/2022-10-24_18/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_2M_全_多方向_误报_1024/2022-10-24_18/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_2M_全_多方向_误报_1024/2022-10-24_19/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_2M_全_多方向_误报_1024/2022-10-24_19/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_2M_全_多方向_误报_1115/2022-11-15_12_16/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_2M_全_多方向_误报_1115/2022-11-15_12_16/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_2M_全_多方向_误报_1115/2022-11-16_07/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_2M_全_多方向_误报_1115/2022-11-16_07/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_2M_全_多方向_误报_1115/2022-11-16_06/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_2M_全_多方向_误报_1115/2022-11-16_06/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//5M_2M_全_多方向_误报_1115/2022-11-16_00_01/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_2M_全_多方向_误报_1115/2022-11-16_00_01/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_C27_anpr/test_video/ZD/ZD_DUBAI/avi文件//test"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/test/"

    args.suffix = '.avi'
    # args.suffix = '.mp4'

    # 是否保存视频结果
    args.write_result_video_bool = True
    # 是否保存每一帧结果
    args.write_result_per_frame_bool = True
    # args.write_result_per_frame_bool = False
    # 是否保存抓拍结果
    args.write_capture_crop_bool = True
    # 是否保存日志
    args.write_csv_bool = True
    # 是否保存原始图像
    args.write_ori_jpg_bool = False
    # 是否保存预测结果图像
    args.write_result_jpg_bool = False

    inference_video(args)


if __name__ == '__main__':
    main()