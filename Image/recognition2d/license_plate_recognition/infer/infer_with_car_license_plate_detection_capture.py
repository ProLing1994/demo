import argparse
from collections import Counter
import cv2
import numpy as np
import os
import sys 
import random
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
# sys.path.insert(0, '/yuanhuan/code/demo/Image')
from detection2d.ssd_rfb_crossdatatraining.test_tools import SSDDetector
from regreesion2d.plate_regreesion.utils.draw_tools import draw_detection_result
from recognition2d.license_plate_recognition.infer.license_plate import license_palte_model_init_caffe, license_palte_crnn_recognition_caffe, license_palte_beamsearch_init, license_palte_crnn_recognition_beamsearch_caffe


sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *


def check_in_roi(in_box, roi_bbox):
    roi_bool = False

    if in_box[0] >= roi_bbox[0] and in_box[2] <= roi_bbox[2] and in_box[1] >= roi_bbox[1] and in_box[3] <= roi_bbox[3]:
        roi_bool = True

    return roi_bool


def intersect(box_a, box_b):
    inter_x1 = max(box_a[0], box_b[0])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y1 = max(box_a[1], box_b[1])
    inter_y2 = min(box_a[3], box_b[3])
    inter =  max(inter_x2 - inter_x1, 0.0) * max(inter_y2 - inter_y1, 0.0) 
    return inter


def get_edit_distance(sentence1, sentence2):
    '''
    :param sentence1: sentence1 list
    :param sentence2: sentence2 list
    :return: distence between sentence1 and sentence2
    '''
    len1 = len(sentence1)
    len2 = len(sentence2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if sentence1[i-1] == sentence2[j-1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta,
                           min(dp[i-1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


def model_init(args):
    # model init
    car_plate_detector = SSDDetector(prototxt=args.ssd_car_plate_prototxt, model_path=args.ssd_car_plate_model_path, ssd_caffe_bool=args.ssd_caffe_bool, merge_class_bool=args.merge_class_bool)
    license_palte_detector = license_palte_model_init_caffe(args.plate_recognition_prototxt, args.plate_recognition_model_path)
    license_palte_beamsearch = license_palte_beamsearch_init()
    return (car_plate_detector, license_palte_detector, license_palte_beamsearch)


def img_detect(args, model, img):
    car_plate_detector = model[0]
    license_palte_detector = model[1]
    license_palte_beamsearch = model[2]

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # init
    show_bboxes = {}
    license_palte_ocr_list = []
    capture_bool = False
    
    # capture roi
    show_bboxes["roi_capture_area"] = [args.roi_capture_area]

    # car_plate_detector
    bboxes = car_plate_detector.detect(img)
    
    # license_plate
    for plate_idx in range(len(bboxes["license_plate"])):
        plate_bbox = bboxes["license_plate"][plate_idx]

        roi_bool = check_in_roi(plate_bbox, args.roi_capture_area)
        if not roi_bool:
            continue
        
        # find car bbox
        for car_idx in range(len(bboxes["car"])):
            car_bbox = bboxes["car"][car_idx]
            if intersect(plate_bbox, car_bbox) > 0.0:
                if "car" not in show_bboxes:
                    show_bboxes["car"] = [car_bbox]    
                else:
                    show_bboxes["car"].append(car_bbox)

        # crop
        crop_img = gray_img[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]
        
        # check
        if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
            continue

        if args.prefix_beam_search_bool:
            # prefix beamsearch
            _, result_scors_list = license_palte_crnn_recognition_caffe(license_palte_detector, crop_img)
            result_ocr = license_palte_crnn_recognition_beamsearch_caffe(license_palte_detector, crop_img, license_palte_beamsearch[0], license_palte_beamsearch[1])
        else:
            # greedy
            result_ocr, result_scors_list = license_palte_crnn_recognition_caffe(license_palte_detector, crop_img)

        # capture bool
        plate_center_y = (plate_bbox[1] + plate_bbox[3]) / 2
        distance_y = max( args.roi_capture_area[3] - plate_center_y, 0.0 )
        if distance_y <= args.capture_min_thres * ( args.roi_capture_area[3] - args.roi_capture_area[1]):
            capture_bool = True

        show_bboxes[result_ocr] = [plate_bbox]
        license_palte_ocr_list.append(result_ocr)

    return show_bboxes, license_palte_ocr_list, capture_bool


def inference_vidio(args):
    # mkdir 
    if args.write_bool:
        create_folder(args.output_vidio_dir)

    # model init
    model = model_init(args)

    # image init 
    vidio_list = np.array(os.listdir(args.vidio_dir))
    vidio_list = vidio_list[[vidio.endswith('.avi') for vidio in vidio_list]]
    vidio_list.sort()
    
    for idx in tqdm(range(len(vidio_list))):
        vidio_path = os.path.join(args.vidio_dir, vidio_list[idx])
       
        cap = cv2.VideoCapture(vidio_path) 
        print(int(cap.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧

        if args.write_bool:
            output_vidio_path = os.path.join(args.output_vidio_dir, vidio_list[idx].replace('.avi', ''), vidio_list[idx])
            create_folder(os.path.dirname(output_vidio_path))

            # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            # video_writer = cv2.VideoWriter(output_vidio_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        # init
        frame_idx = 0
        capture_idx = 0
        capture_list_per_interval = []
        info_list_per_frame = []
        been_captured_license_palte_list = []

        while True:
            ret, img = cap.read()

            if not ret: # if the camera over return false
                # video_writer.release()
                break
            
            # if frame_idx == 152:
            #     print()

            # detect 
            show_bboxes, license_palte_ocr_list, capture_bool = img_detect(args, model, img)

            # capture_bool 
            ## 显示叠加字符，便于可视化
            if capture_bool:
                if 'car' in show_bboxes:
                    show_bboxes['car_capture'] = show_bboxes['car']

            # draw
            img = draw_detection_result(img, show_bboxes, mode='ltrb')

            # capture
            # 每一帧存储车牌结果信息
            info_list_per_frame.append(license_palte_ocr_list)

            # 跳帧存储原图和检测识别结果
            if frame_idx % args.capture_interval == 0:
                capture_list_per_interval.append({'image': img, 'bboxes': show_bboxes})

            if len(info_list_per_frame) >= args.info_container:
                info_list_per_frame.pop(0)

            if len(capture_list_per_interval) > args.capture_container:
                capture_list_per_interval.pop(0)

            # draw img
            if args.write_bool:
                # 保存视频结果
                # video_writer.write(img)

                # 保存图像结果
                output_img_path = os.path.join(args.output_vidio_dir, vidio_list[idx].replace('.avi', ''), vidio_list[idx].replace('.avi', '_{}.jpg'.format(frame_idx)))
                create_folder(os.path.dirname(output_img_path))
                cv2.imwrite(output_img_path, img)

                # capture_bool
                if capture_bool:
                    info_license_palte_list = [info_list_per_frame[idx][idy] for idx in range(len(info_list_per_frame)) if len(info_list_per_frame[idx]) for idy in range(len(info_list_per_frame[idx]))]
                    info_license_palte = Counter(info_license_palte_list).most_common(1)[0][0]

                    if info_license_palte in been_captured_license_palte_list:
                        continue
                    else:
                        been_captured_license_palte_list.append(info_license_palte)

                    select_capture_list = [capture_list_per_interval[idx] for idx in range(len(capture_list_per_interval)) if info_license_palte in capture_list_per_interval[idx]['bboxes']]

                    if len(select_capture_list) == 0 and args.capture_edit_distance_bool:
                        # 利用编辑距离思想，挑选近似结果
                        for idy in range(len(capture_list_per_interval)):
                            for key in capture_list_per_interval[idy]['bboxes'].keys():
                                edit_distance = get_edit_distance(info_license_palte, key)
                                if edit_distance <= args.capture_edit_distance_thres:
                                    select_capture_list.append(capture_list_per_interval[idy])

                    if len(select_capture_list) > 3:
                        select_capture_list = random.sample(select_capture_list, 3)

                    for idy in range(len(select_capture_list)):
                        # 保存捕获结果
                        output_capture_path = os.path.join(args.output_vidio_dir, vidio_list[idx].replace('.avi', ''), 'capture', vidio_list[idx].replace('.avi', '_{}.jpg'.format(capture_idx)))
                        create_folder(os.path.dirname(output_capture_path))
                        cv2.imwrite(output_capture_path, select_capture_list[idy]['image'])

                        capture_idx += 1

                frame_idx += 1

                tqdm.write("{}: {}".format(vidio_path, str(frame_idx)))


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.ssd_car_plate_prototxt = None
    args.ssd_car_plate_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-02-24-15_focalloss_4class_car_bus_truck_licenseplate_zg_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
    # args.ssd_car_plate_prototxt = "/mnt/huanyuan2/model/image_model/ssd_rfb_zg/FPN_RFB_5class_noDilation_prior.prototxt"
    # args.ssd_car_plate_model_path = "/mnt/huanyuan2/model/image_model/ssd_rfb_zg/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_zg_2022-02-24-15.caffemodel"
    args.ssd_caffe_bool = False

    args.plate_recognition_prototxt = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china_softmax.prototxt"
    args.plate_recognition_model_path = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china.caffemodel"
    args.prefix_beam_search_bool = True

    # 是否保存结果
    args.write_bool = True

    # 对 roi 区域进行抓拍
    args.roi_capture_area = [900, 0, 1900, 1920]

    # 抓拍阈值，距离下边界的距离
    args.capture_min_thres = 0.10

    # 抓拍间隔
    args.capture_interval = 2

    # 抓拍图像匹配编辑距离
    args.capture_edit_distance_bool = True
    args.capture_edit_distance_thres = 1.0

    # 抓拍容器
    args.capture_container = 8

    # 信息容器
    args.info_container = 30

    # 是否将 car\bus\truck 合并为一类输出
    args.merge_class_bool = True

    args.vidio_bool = True
    args.vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频/test/"
    args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频/test_capture/"
    # args.vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频/5M/"
    # args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频/5M_capture/"

    if args.vidio_bool:
        inference_vidio(args)


if __name__ == '__main__':
    main()