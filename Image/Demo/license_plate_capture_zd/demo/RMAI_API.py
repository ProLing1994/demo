from collections import Counter
import cv2
import numpy as np
import sys 
import random

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.detection2d.ssd_rfb_crossdatatraining.test_tools import SSDDetector
from Image.detection2d.mmdetection.demo.detector.yolov6_detector import YOLOV6Detector
from Image.Demo.license_plate_capture_zd.sort.mot_sort import Sort


def intersect(box_a, box_b):
    inter_x1 = max(box_a[0], box_b[0])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y1 = max(box_a[1], box_b[1])
    inter_y2 = min(box_a[3], box_b[3])
    inter =  max(inter_x2 - inter_x1, 0.0) * max(inter_y2 - inter_y1, 0.0) 
    return inter


def bool_box_in_roi(box, roi):
    bool_in_w = True if box[0] >= roi[0] and box[2] <= roi[2] else False
    bool_in_h = True if box[1] >= roi[1] and box[3] <= roi[3] else False
    return bool_in_w * bool_in_h


class CaptureApi():
    """
    CaptureApi
    """

    def __init__(self):

        # option
        self.option_init()

        # param_init
        self.param_init()

        # model_init 
        self.model_init()


    def option_init(self):

        self.image_width = 2592
        self.image_height = 1920

        # ssd
        self.ssd_bool = True
        self.ssd_caffe_bool = False
        self.ssd_openvino_bool = False
        # 2022-08-10-00
        # pytorch 
        self.ssd_car_plate_prototxt = None
        self.ssd_car_plate_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-08-10-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_zf_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
        # # # caffe
        # # self.ssd_car_plate_prototxt = ""
        # # self.ssd_car_plate_model_path = ""
        # # openvino
        # # self.ssd_car_plate_prototxt = None
        # # self.ssd_car_plate_model_path = ""

        # yolov6
        self.yolov6_bool = False
        # pytorch
        self.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_zg_data_0722/yolov6_zg_bmx.py"
        self.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_zg_data_0722/epoch_300.pth"

        # 是否将 car\bus\truck 合并为一类输出
        self.merge_class_bool = True
        self.merge_class_name = 'car_bus_truck'
        self.car_attri_name_list = [ 'car', 'bus', 'truck' ]
        self.license_plate_name = 'license_plate'

        # self.detect_class_name = ['car_bus_truck', 'non_motorized', 'person']
        # self.detect_class_threshold_list = [0.5, 0.4, 0.4]

        # sort
        self.max_age = 10
        self.min_hits = 3 
        self.iou_threshold = 0.3
    

    def param_init(self):
        self.params_dict = {}

        # bbox_info_dict
        bbox_info_dict = {}
        bbox_info_dict['id'] = 0                                            # 追踪id
        bbox_info_dict['loc'] = []                                          # 检测框坐标
        bbox_info_dict['plate_loc'] = []                                    # 车牌坐标


    def model_init(self):
        
        # detector
        if self.ssd_bool:
            self.detector = SSDDetector(prototxt=self.ssd_car_plate_prototxt, model_path=self.ssd_car_plate_model_path, ssd_caffe_bool=self.ssd_caffe_bool, ssd_openvino_bool=self.ssd_openvino_bool, merge_class_bool=self.merge_class_bool)

        elif self.yolov6_bool:
            self.detector = YOLOV6Detector(self.yolov6_config, self.yolov6_checkpoint, class_name=self.detect_class_name, threshold_list=self.detect_class_threshold_list)

        # tracker
        self.mot_tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_threshold)


    def clear(self):

        # param_init
        self.param_init()


    def run(self, img, frame_idx):

        # info 
        image_width = img.shape[1]
        image_height = img.shape[0]

        assert self.image_width == image_width
        assert self.image_height == image_height

        # detector
        bboxes = self.detector.detect( img, with_score=True )
    
        # tracker 
        tracker_bboxes = self.update_tracker_bboxes( bboxes )

        # update bbox info
        bbox_info_list = self.update_bbox_info( img, bboxes, tracker_bboxes )

        return tracker_bboxes, bbox_info_list


    def update_tracker_bboxes(self, bboxes):
    
        if self.merge_class_bool:
            # tracker
            if self.merge_class_name in bboxes:
                dets = np.array(bboxes[self.merge_class_name])
            else:
                dets = np.empty((0, 5))
            tracker_bboxes = self.mot_tracker.update(dets)

        else:
            # tracker
            dets = np.empty((0, 5))
            for idx in range(len(self.car_attri_name_list)):
                car_attri_name_idx = self.car_attri_name_list[idx]
                if car_attri_name_idx in bboxes:
                    dets = np.concatenate((dets, np.array(bboxes[car_attri_name_idx])), axis=0)
                    
            tracker_bboxes = self.mot_tracker.update(dets)

        return tracker_bboxes


    def match_bbox_iou(self, input_roi, match_roi_list):
        # init
        matched_roi_list = []
        max_intersect_iou = 0.0
        max_intersect_iou_idx = 0

        for idx in range(len(match_roi_list)):
            match_roi_idx = match_roi_list[idx][0:4]
            intersect_iou = intersect(input_roi, match_roi_idx)

            if intersect_iou > max_intersect_iou:
                max_intersect_iou = intersect_iou
                max_intersect_iou_idx = idx
            
        if max_intersect_iou > 0.0:
            matched_roi_list.append(match_roi_list[max_intersect_iou_idx])
        
        return matched_roi_list


    def match_car_license_plate(self, car_roi, license_plate_list):

        # sort_key
        def sort_key(data):
            return data[-1]

        # init
        matched_roi_list = []

        for idx in range(len(license_plate_list)):
            match_roi_idx = license_plate_list[idx][0:4]

            # 方案一：计算车牌框完全位于车框内
            bool_in = bool_box_in_roi(match_roi_idx, car_roi)
            if bool_in:
                matched_roi_list.append(license_plate_list[idx])

        matched_roi_list.sort(key=sort_key, reverse=True)

        return matched_roi_list


    def update_bbox_info(self, img, bboxes, tracker_bboxes):
        
        bbox_info_list = []
        for idx in range(len(tracker_bboxes)):
            # init 
            # bbox_info_dict
            bbox_info_dict = {}
            bbox_info_dict['id'] = 0                                            # 追踪id
            bbox_info_dict['loc'] = []                                          # 检测框坐标
            bbox_info_dict['plate_loc'] = []                                    # 车牌坐标

            # car
            tracker_bbox = tracker_bboxes[idx]
            bbox_info_dict['id'] = tracker_bbox[-1]
            bbox_info_dict['loc'] = tracker_bbox[0:4]

            # license plate
            if self.license_plate_name in bboxes:

                license_plate_roi_list = bboxes[self.license_plate_name]
                # 求相交同时置信度最高的车牌框
                match_license_plate_roi = self.match_car_license_plate(bbox_info_dict['loc'], license_plate_roi_list)

                if len(match_license_plate_roi):
                    bbox_info_dict['plate_loc'] = match_license_plate_roi[0][0:4]

            bbox_info_list.append(bbox_info_dict)

        return bbox_info_list