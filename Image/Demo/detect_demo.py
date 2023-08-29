import argparse
import cv2
import numpy as np
import os
import sys
import time 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
# from Image.Basic.utils.folder_tools import *
from Image.detection2d.ssd_rfb_crossdatatraining.test_tools import SSDDetector
from Image.detection2d.mmdetection.demo.detector.yolov6_detector import YOLOV6Detector


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def inference_img(args):
    
    # mkdir 
    create_folder(args.output_img_dir)

    # model init 
    detector = YOLOV6Detector(args.yolov6_config, args.yolov6_checkpoint, class_name=args.yolov6_class_name, threshold_list=args.yolov6_threshold_list, device='cuda:0')

    # img init 
    img_list = np.array(os.listdir(args.img_dir))
    img_list = img_list[[img.endswith(args.suffix) for img in img_list]]
    img_list.sort()

    for idx in tqdm(range(len(img_list))):

        img_path = os.path.join(args.img_dir, img_list[idx])
        output_img_path = os.path.join(args.output_img_dir, img_list[idx])

        # img
        img = cv2.imread(img_path)

        # segmentor
        start = time.time()
        bboxes = detector.detect( img, with_score=True )
        end = time.time()
        print('Running time: %s Seconds'%(end-start))

        for key in bboxes:
            for idx in range(len(bboxes[key])):
                roi_idx = bboxes[key][idx][0:4]
                score_idx = bboxes[key][idx][-1]

                cv2.rectangle(img, (int(roi_idx[0]), int(roi_idx[1])), (int(roi_idx[2]), int(roi_idx[3])), color=(0, 0, 255), thickness=2)
                img = cv2.putText(img, "{}_{:.2f}".format( key, score_idx), (int(roi_idx[0]), int(roi_idx[3]) + 30), 
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(output_img_path, img)



def inference_video(args):

    # mkdir 
    create_folder(args.output_video_dir)

    # model init 
    detector = SSDDetector(prototxt=args.ssd_prototxt, model_path=args.ssd_model_path, ssd_caffe_bool=True, ssd_openvino_bool=False, merge_class_bool=False, gpu_bool=True)
    # detector = YOLOV6Detector(args.yolox_config, args.yolox_checkpoint, class_name=args.yolox_class_name, threshold_list=args.yolox_threshold_list, device='cuda:0')

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
        output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), True)

        while True:
            ret, img = cap.read()

            if not ret: # if the camera over return false
                break

            # detector
            start = time.time()
            bboxes = detector.detect( img, with_score=True )
            end = time.time()
            print('Running time: %s Seconds'%(end-start))

            for key in bboxes:
                for idx in range(len(bboxes[key])):
                    roi_idx = bboxes[key][idx][0:4]
                    score_idx = bboxes[key][idx][-1]

                    cv2.rectangle(img, (int(roi_idx[0]), int(roi_idx[1])), (int(roi_idx[2]), int(roi_idx[3])), color=(0, 0, 255), thickness=2)
                    img = cv2.putText(img, "{}_{:.2f}".format( key, score_idx), (int(roi_idx[0]), int(roi_idx[3]) + 30), 
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    
            output_video.write(img)

            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # bm car
    args.img_dir = "/mnt/huanyuan2/data/image/RM_C28_detection/test_img/"
    args.output_img_dir = "/mnt/huanyuan2/data/image/RM_C28_detection/test_img_res/"

    args.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0320/yolov6_rm_c28_deploy.py"
    args.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0320/epoch_340_deploy.pth"
    args.yolov6_class_name = ['car']
    args.yolov6_threshold_list = [0.3]
    
    args.suffix = '.jpg'
    inference_img(args)

    # # r151
    # # args.video_dir = "/mnt/huanyuan2/data/image/test_R151_detect/car_avi_select/车前/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/test_R151_detect/yolox_140/车前/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/test_R151_detect/car_avi_select/右侧/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/test_R151_detect/yolox_140/右侧/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/test_R151_detect/car_avi_select/右俯视/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/test_R151_detect/yolox_140/右俯视/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/test_R151_detect/car_avi_select/左侧/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/test_R151_detect/yolox_140/左侧/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/test_R151_detect/car_avi_select/左俯视/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/test_R151_detect/yolox_140/左俯视/"
    
    # # Brazil
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_BM_Brazil_C27/5M_白天_2022_1026/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_BM_Brazil_C27/5M_白天_2022_1026_ssd_rfb_0609/"

    # # # ZD_DUBAI
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_白天_侧向_0615/avi/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZD_DUBAI/5M_白天_侧向_0615/yolox_0609/"

    # # zg
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/5M_230416/avi/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/ssd_rfb_0609/"

    # # ssd caffe
    # # args.ssd_prototxt = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-08-10-00/FPN_RFB_3class_3attri_noDilation_prior.prototxt"
    # # args.ssd_model_path = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-08-10-00/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-08-10-00.caffemodel"

    # # SSD_VGG_FPN_RFB_2023-06-09_focalloss_5class_car_bus_truck_motorcyclist_licenseplate_softmax
    # args.ssd_prototxt = "/mnt/huanyuan/model_final/image_model/schoolbus/ssd_rfb/SSD_VGG_FPN_RFB_2023-06-09_focalloss_5class_car_bus_truck_motorcyclist_licenseplate_softmax/FPN_RFB_4class_3attri_noDilation_prior.prototxt"
    # args.ssd_model_path = "/mnt/huanyuan/model_final/image_model/schoolbus/ssd_rfb/SSD_VGG_FPN_RFB_2023-06-09_focalloss_5class_car_bus_truck_motorcyclist_licenseplate_softmax/SSD_VGG_FPN_RFB_VOC_car_bus_truck_motorcyclist_licenseplate_2023_06_09_70.caffemodel"

    # # # yolox pth
    # # args.yolox_config = "/mnt/huanyuan/model/image/yolox/yolovx_l_license_0601/yolox_l_license.py"
    # # args.yolox_checkpoint =  "/mnt/huanyuan/model/image/yolox/yolovx_l_license_0601/epoch_140.pth"
    # # args.yolox_class_name = ['license_plate']
    # # args.yolox_threshold_list = [0.4]

    # args.suffix = '.avi'
    # inference_video(args)


if __name__ == '__main__':
    main()