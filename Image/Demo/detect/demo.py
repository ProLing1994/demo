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
from Image.Demo.license_plate_capture_Brazil_police.model.LPR_detect import LPRDetectCaffe, LPRDetectOpenVINO
from Image.detection2d.mmdetection.demo.detector.yolov6_landmark_detector import YOLOV6LandmarkDetector


def inference_car_video(args):

    # mkdir 
    create_folder(args.output_video_dir)

    # model init 
    detector = SSDDetector(prototxt=args.ssd_prototxt, model_path=args.ssd_model_path, ssd_caffe_bool=True, ssd_openvino_bool=False, merge_class_bool=False, gpu_bool=True)
    # detector = LPRDetectCaffe(args.ssd_prototxt, args.ssd_model_path, class_name=args.detect_class_name, gpu_bool=True, conf_thres=args.ssd_conf_thres)

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

            if 'license_plate' in bboxes:
                for idx in range(len(bboxes['license_plate'])):
                    roi_idx = bboxes['license_plate'][idx][0:4]

                    cv2.rectangle(img, (roi_idx[0], roi_idx[1]), (roi_idx[2], roi_idx[3]), color=(0, 0, 255), thickness=2)

            output_video.write(img)

            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))


def inference_face_video(args):
    
    # mkdir 
    create_folder(args.output_video_dir)

    # model init 
    detector = YOLOV6LandmarkDetector(args.yolov6_config, args.yolov6_checkpoint, class_name=args.yolov6_class_name, threshold_list=args.yolov6_threshold_list, device='cuda:0')

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

            if 'face' in bboxes:
                for idx in range(len(bboxes['face'])):
                    roi_idx = bboxes['face'][idx][0:4]
                    landmark_idx = bboxes['face'][idx][4:14]

                    cv2.rectangle(img, (roi_idx[0], roi_idx[1]), (roi_idx[2], roi_idx[3]), color=(0, 0, 255), thickness=2)
                    # cv2.circle(img, (int(landmark_idx[0] + 0.5), int(landmark_idx[1] + 0.5)), 3, (0, 0, 255), 2)
                    # cv2.circle(img, (int(landmark_idx[2] + 0.5), int(landmark_idx[3] + 0.5)), 3, (0, 0, 255), 2)
                    # cv2.circle(img, (int(landmark_idx[4] + 0.5), int(landmark_idx[5] + 0.5)), 3, (0, 0, 255), 2)
                    # cv2.circle(img, (int(landmark_idx[6] + 0.5), int(landmark_idx[7] + 0.5)), 3, (0, 0, 255), 2)
                    # cv2.circle(img, (int(landmark_idx[8] + 0.5), int(landmark_idx[9] + 0.5)), 3, (0, 0, 255), 2)

            output_video.write(img)

            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # car
    # args.video_dir = "/mnt/huanyuan2/data/image/test_R151_detect/avi/左俯视/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/test_R151_detect/左俯视/"

    # args.ssd_prototxt = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-08-10-00/FPN_RFB_3class_3attri_noDilation_prior.prototxt"
    # args.ssd_model_path = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-08-10-00/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-08-10-00.caffemodel"

    # # args.ssd_prototxt = "/mnt/huanyuan/model_final/image_model/schoolbus/zd_ssd_rfb_wmr/ssd_mbv2_2class/caffe_model/ssd_mobilenetv2_fpn.prototxt"
    # # args.ssd_model_path = "/mnt/huanyuan/model_final/image_model/schoolbus/zd_ssd_rfb_wmr/ssd_mbv2_2class/caffe_model/ssd_mobilenetv2_0421.caffemodel"
    # # args.detect_class_name = ['license_plate']
    # # args.ssd_conf_thres = 0.25

    # args.suffix = '.avi'
    # inference_car_video(args)

    # face
    args.video_dir = "/mnt/huanyuan2/data/image/test_R151_detect/face_avi/左俯视/"
    args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/test_R151_detect/face_avi/左俯视/"

    # yolov6 landmark
    args.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_20230526/yolov6_face_wider_face.py"
    args.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_20230526/epoch_400.pth"
    args.yolov6_class_name = ['face']
    args.yolov6_threshold_list = [0.4]

    args.suffix = '.avi'
    inference_face_video(args)


if __name__ == '__main__':
    main()