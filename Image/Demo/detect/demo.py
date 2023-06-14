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
from Image.detection2d.ssd_rfb_crossdatatraining.test_tools import SSDDetector
from Image.detection2d.mmdetection.demo.detector.yolov6_detector import YOLOV6Detector
from Image.detection2d.mmdetection.demo.detector.yolov6_landmark_detector import YOLOV6LandmarkDetector


def inference_video(args):

    # mkdir 
    create_folder(args.output_video_dir)

    # model init 
    # detector = SSDDetector(prototxt=args.ssd_prototxt, model_path=args.ssd_model_path, ssd_caffe_bool=True, ssd_openvino_bool=False, merge_class_bool=False, gpu_bool=True)
    detector = YOLOV6Detector(args.yolox_config, args.yolox_checkpoint, class_name=args.yolox_class_name, threshold_list=args.yolox_threshold_list, device='cuda:0')

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

        output_video_path = os.path.join(args.output_video_dir, video_list[idx])
        create_folder(os.path.dirname(output_video_path))

        frame_idx = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (1920, 1080), True)

        while True:
            ret, img = cap.read()

            if not ret: # if the camera over return false
                break

            # detector
            bboxes = detector.detect( img, with_score=True )

            if args.detect_class_name in bboxes:
                for idx in range(len(bboxes[args.detect_class_name])):
                    roi_idx = bboxes[args.detect_class_name][idx][0:4]
                    score_idx = bboxes[args.detect_class_name][idx][-1]

                    cv2.rectangle(img, (roi_idx[0], roi_idx[1]), (roi_idx[2], roi_idx[3]), color=(0, 0, 255), thickness=2)
                    img = cv2.putText(img, "{:.2f}".format( score_idx), (roi_idx[0], roi_idx[3] + 30), 
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    
            output_video.write(img)

            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.video_dir = "/mnt/huanyuan2/data/image/test_R151_detect/car_avi_select/车前/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/test_R151_detect/yolox/车前/"
    # args.video_dir = "/mnt/huanyuan2/data/image/test_R151_detect/car_avi_select/右侧/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/test_R151_detect/yolox/右侧/"
    # args.video_dir = "/mnt/huanyuan2/data/image/test_R151_detect/car_avi_select/右俯视/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/test_R151_detect/yolox/右俯视/"
    # args.video_dir = "/mnt/huanyuan2/data/image/test_R151_detect/car_avi_select/左侧/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/test_R151_detect/yolox/左侧/"
    args.video_dir = "/mnt/huanyuan2/data/image/test_R151_detect/car_avi_select/左俯视/"
    args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/test_R151_detect/yolox/左俯视/"

    # ssd caffe
    # args.ssd_prototxt = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-08-10-00/FPN_RFB_3class_3attri_noDilation_prior.prototxt"
    # args.ssd_model_path = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-08-10-00/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-08-10-00.caffemodel"

    # yolox pth
    args.yolox_config = "/mnt/huanyuan/model/image/yolox/yolovx_l_license_0601/yolox_l_license.py"
    args.yolox_checkpoint =  "/mnt/huanyuan/model/image/yolox/yolovx_l_license_0601/epoch_75.pth"
    args.yolox_class_name = ['license_plate']
    args.yolox_threshold_list = [0.4]

    args.suffix = '.avi'
    args.detect_class_name = 'license_plate'
    inference_video(args)


if __name__ == '__main__':
    main()