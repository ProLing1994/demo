from collections import Counter
import cv2
import copy
import numpy as np
import sys 
import random
import xml.etree.ElementTree as ET

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.detection2d.ssd_rfb_crossdatatraining.test_tools import SSDDetector
from Image.detection2d.mmdetection.demo.detector.yolov6_detector import YOLOV6Detector

from Image.recognition2d.lpr.infer.lpr import LPRCaffe, LPRPytorch
from Image.recognition2d.lpr.infer.lpr_seg import LPRSegColorClassCaffe

from Image.Demo.bus_occupying_capture.sort.mot_sort import Sort

from Image.Demo.bus_occupying_capture.info.options_lpr_china import options
from Image.Demo.bus_occupying_capture.info.param import *
from Image.Demo.bus_occupying_capture.utils.iou import *
from Image.Demo.bus_occupying_capture.utils.landmark2degree import landmark2degree

import Image.recognition2d.script.paddle.infer.lpr as paddle_lpr


def car_in_roi(bbox_state_idy):
    # 标定线纵向平行线
    # y = kx + b
    capture_point_top = options.roi_lane_line_points[1]
    capture_point_bottom = options.roi_lane_line_points[2]
    capture_rigrt_line_k = float(capture_point_top[1] - capture_point_bottom[1]) / float(capture_point_top[0] - capture_point_bottom[0] + 1e-5);
    capture_right_line_b = float(capture_point_top[1] - capture_point_top[0] * capture_rigrt_line_k);

    capture_point_top = options.roi_lane_line_points[0]
    capture_point_bottom = options.roi_lane_line_points[3]
    capture_left_line_k = float(capture_point_top[1] - capture_point_bottom[1]) / float(capture_point_top[0] - capture_point_bottom[0] + 1e-5);
    capture_left_line_b = float(capture_point_top[1] - capture_point_top[0] * capture_left_line_k);

    # 车辆坐标
    if bbox_state_idy['plate_info']['num'] != '':
        car_center_x = int((bbox_state_idy['plate_info']['roi'][0] + bbox_state_idy['plate_info']['roi'][2]) / 2)
        car_center_y = int((bbox_state_idy['plate_info']['roi'][1] + bbox_state_idy['plate_info']['roi'][3]) / 2)
    else:
        car_center_x = int((bbox_state_idy['car_info']['roi'][0] + bbox_state_idy['car_info']['roi'][2]) / 2)
        car_center_y = int((bbox_state_idy['car_info']['roi'][1] + bbox_state_idy['car_info']['roi'][3]) / 2)

    alarm_flage = False
    for idx in range(int(len(options.roi_lane_line_points) / 2)):
        # 标定线
        point_1 = tuple( options.roi_lane_line_points[ 2 * idx] )
        point_2 = tuple( options.roi_lane_line_points[ 2 * idx + 1] ) 
        point_center_x = int((point_1[0] + point_2[0]) / 2)
        # y = kx + b
        point_line_k = float(point_1[1] - point_2[1]) / float(point_1[0] - point_2[0] + 1e-5);
        point_line_b = float(point_1[1] - point_1[0] * point_line_k);

        # 车辆中心点纵向平行线
        # riget
        if car_center_x > point_center_x:
            car_line_k = capture_rigrt_line_k
            car_line_b = car_center_y - car_line_k * car_center_x
        # left
        else:
            car_line_k = capture_left_line_k
            car_line_b = car_center_y - car_line_k * car_center_x
        car_line = [car_line_k, car_line_b]

        # 判断标定线和车辆中心点纵向平行线 是否相交
        point_intersect_x = (car_line_b - point_line_b)/(point_line_k - car_line_k)
        point_intersect_y = point_line_k * point_intersect_x + point_line_b
        intersect_point = [point_intersect_x, point_intersect_y]

        if point_intersect_x >= min(point_1[0], point_2[0]) and point_intersect_x <= max(point_1[0], point_2[0]) and \
            point_intersect_y >= min(point_1[1], point_2[1]) and point_intersect_y <= max(point_1[1], point_2[1]) and \
            point_intersect_x >= bbox_state_idy['car_info']['roi'][0] and point_intersect_x <= bbox_state_idy['car_info']['roi'][2] and \
            point_intersect_y >= bbox_state_idy['car_info']['roi'][1] and point_intersect_y <= bbox_state_idy['car_info']['roi'][3]:

            alarm_flage = True
            break
    
    return alarm_flage, car_line, intersect_point

class CaptureApi():
    """
    CaptureApi
    """

    def __init__(self, demo_type, country_type):
        
        self.demo_type = demo_type
        self.country_type = country_type

        # option
        self.option_init()

        # param_init
        self.param_init()

        # model_init 
        self.model_init()
    

    def option_init(self):

        self.options = options


    def param_init(self):
        self.params_dict = {}
        
        self.params_dict['bbox_info_container'] = []                        # 缓存容器（imag & bbox_info_dict）
        self.params_dict['bbox_state_container'] = {}                       # 状态信息容器（key: 追踪id, value: bbox_state_dict）
        self.params_dict['capture_container'] = {}                          # 抓拍序列
        self.params_dict['capture_res_container'] = {}                      # 抓拍结果
        self.params_dict['capture_occupying_container'] = {}                # 抓拍车道占道序列
        self.params_dict['capture_occupying_res_container'] = {}            # 抓拍车道占道结果


    def model_init(self):

        if self.demo_type == "lpr":
            # detector
            if options.ssd_bool:
                if self.country_type == "china":
                    self.detector = SSDDetector(prototxt=options.ssd_prototxt, model_path=options.ssd_model_path, ssd_caffe_bool=options.ssd_caffe_bool, ssd_openvino_bool=options.ssd_openvino_bool, merge_class_bool=options.car_attri_merge_bool, gpu_bool=True)
            elif options.yolov6_bool:
                self.detector = YOLOV6Detector(options.yolov6_config, options.yolov6_checkpoint, class_name=options.yolov6_class_name, threshold_list=options.yolov6_threshold_list, device=options.device)

            # lincense plate reader
            if self.country_type == "china":
                if options.lpr_caffe_bool:
                    if options.lpr_paddle_bool: 
                        self.lpr = paddle_lpr.LPRCaffe(options.china.ocr_caffe_model_path, options.china.ocr_caffe_prototxt, options.china.ocr_labels_dict_path, img_shape=options.china.input_shape, padding_bool=options.china.padding_bool, gpu_bool=options.gpu_bool)
                    else:
                        self.lpr = LPRCaffe(options.china.ocr_caffe_prototxt, options.china.ocr_caffe_model_path, input_shape=options.china.input_shape, ocr_labels=options.china.ocr_labels, padding_bool=options.china.padding_bool, prefix_beam_search_bool=options.china.ocr_prefix_beam_search_bool, gpu_bool=options.gpu_bool)
                    self.lpr_seg = LPRSegColorClassCaffe(options.china.seg_caffe_prototxt, options.china.seg_caffe_model_path, options.china.seg_city_dict_name, options.china.seg_color_dict_name, input_shape=options.china.seg_input_shape, city_bool=options.china.seg_city_bool, color_bool=options.china.seg_color_bool, gpu_bool=options.gpu_bool)

                elif options.lpr_pytorch_bool:
                    if options.lpr_paddle_bool: 
                        self.lpr = None
                    else:
                        self.lpr = LPRPytorch(options.china.ocr_pth_path, input_shape=options.china.input_shape, ocr_labels=options.china.ocr_labels, prefix_beam_search_bool=options.china.ocr_prefix_beam_search_bool)
                    self.lpr_seg = None

                elif options.lpr_onnx_bool:
                    if options.lpr_paddle_bool: 
                        self.lpr = paddle_lpr.LPROnnx(options.china.ocr_onnx_model_path, options.china.ocr_labels_dict_path, img_shape=options.china.input_shape, padding_bool=options.china.padding_bool, gpu_bool=options.gpu_bool)
                    else:
                        self.lpr = None
                    self.lpr_seg = LPRSegColorClassCaffe(options.china.seg_caffe_prototxt, options.china.seg_caffe_model_path, options.china.seg_city_dict_name, options.china.seg_color_dict_name, input_shape=options.china.seg_input_shape, city_bool=options.china.seg_city_bool, color_bool=options.china.seg_color_bool, gpu_bool=options.gpu_bool)

            # tracker
            self.mot_tracker = Sort(max_age=options.max_age, min_hits=options.min_hits, iou_threshold=options.iou_threshold)


    def clear(self):
        # param_init
        self.param_init()


    def run(self, img, frame_idx):

        # info 
        image_width = img.shape[1]
        image_height = img.shape[0]

        assert options.image_width == image_width
        assert options.image_height == image_height

        # detector
        bboxes = self.detector.detect( img, with_score=True )

        # tracker 
        tracker_bboxes = self.update_tracker_bboxes( bboxes )

        # update bbox info
        bbox_info_list = self.update_bbox_info( img, bboxes, tracker_bboxes )

        # update car info
        bbox_info_list = self.update_car_info( img, bboxes, bbox_info_list )

        # update plate info
        bbox_info_list = self.update_plate_info( img, bboxes, bbox_info_list )

        # store
        # 跳帧存储原图和检测识别结果
        self.update_cache_container( img, frame_idx, bbox_info_list )

        # 更新状态容器，同时更新车辆行驶状态和帧率
        bbox_info_list = self.update_bbox_state_container( bbox_info_list )
        bbox_state_container = self.params_dict['bbox_state_container']

        # 车牌抓拍
        self.update_capture_dict()
        # capture_container = self.params_dict['capture_container']
        self.update_capture_state()
        # capture_res_container = self.params_dict['capture_res_container']

        # 车辆抓拍
        self.update_bbox_state_container_capture_occupying( img, bbox_info_list )
        capture_occupying_container = self.params_dict['capture_occupying_container']
        capture_occupying_res_container = self.params_dict['capture_occupying_res_container']

        ## capture_line
        if not options.roi_lane_line_bool:
            capture_line_points = [(0, options.Up_threshold), (options.image_width, options.Up_threshold), 
                                    (0, options.Down_threshold), (options.image_width, options.Down_threshold),
                                    (options.Left_threshold, 0), (options.Left_threshold, options.image_height),
                                    (options.Right_threshold, 0), (options.Right_threshold, options.image_height)]
        else:
            capture_line_points = [options.roi_lane_line_points[0], options.roi_lane_line_points[1],
                                    options.roi_lane_line_points[1], options.roi_lane_line_points[2],
                                    options.roi_lane_line_points[2], options.roi_lane_line_points[3],
                                    options.roi_lane_line_points[3], options.roi_lane_line_points[0]]

        return tracker_bboxes, bbox_info_list, bbox_state_container, capture_line_points, capture_occupying_container, capture_occupying_res_container
    

    def update_tracker_bboxes(self, bboxes):

        if self.demo_type == "lpr" and options.sort_type == "car":
            if options.car_attri_merge_bool:
                # tracker
                if options.car_attri_merge_name in bboxes:
                    dets = np.array(bboxes[options.car_attri_merge_name])
                else:
                    dets = np.empty((0, 5))
                tracker_bboxes = self.mot_tracker.update(dets)

            else:
                # tracker
                dets = np.empty((0, 5))
                for idx in range(len(options.car_attri_name_list)):
                    car_attri_name_idx = options.car_attri_name_list[idx]
                    if car_attri_name_idx in bboxes:
                        dets = np.concatenate((dets, np.array(bboxes[car_attri_name_idx])), axis=0)
                        
                tracker_bboxes = self.mot_tracker.update(dets)

        elif (self.demo_type == "lpr" and options.sort_type == "plate"):
            # tracker
            dets = np.empty((0, 5))
            for idx in range(len(options.sort_class_name)):
                sort_class_name = options.sort_class_name[idx]
                if sort_class_name in bboxes:

                    # 由于面积太小，跟踪不上，需要增大面积
                    bboxes_class_list = []
                    
                    for idy in range(len(bboxes[sort_class_name])):
                        bboxes_idx = bboxes[sort_class_name][idy]
                        
                        bboxes_idx_width = ( bboxes_idx[2] - bboxes_idx[0] ) * options.sort_expand_ratio
                        bboxes_idx_height = ( bboxes_idx[3] - bboxes_idx[1] ) * options.sort_expand_ratio
                        bboxes_idx[0] = bboxes_idx[0] - bboxes_idx_width
                        bboxes_idx[2] = bboxes_idx[2] + bboxes_idx_width
                        bboxes_idx[1] = bboxes_idx[1] - bboxes_idx_height
                        bboxes_idx[3] = bboxes_idx[3] + bboxes_idx_height
                        bboxes_class_list.append([bboxes_idx[0], bboxes_idx[1], bboxes_idx[2], bboxes_idx[3], bboxes_idx[-1]])

                    dets = np.concatenate((dets, np.array(bboxes_class_list)), axis=0)
            
            tracker_bboxes = self.mot_tracker.update(dets)

            # 由于增加了面积，需要减少面积，获得精确坐标
            for idx in range(len(tracker_bboxes)):
                bboxes_idx = tracker_bboxes[idx]

                bboxes_idx_width = (( bboxes_idx[2] - bboxes_idx[0] ) * options.sort_expand_ratio ) / ( 1.0 + options.sort_expand_ratio * 2.0 )
                bboxes_idx_height = (( bboxes_idx[3] - bboxes_idx[1] ) * options.sort_expand_ratio )  / ( 1.0 + options.sort_expand_ratio * 2.0 )
                bboxes_idx[0] = bboxes_idx[0] + bboxes_idx_width
                bboxes_idx[2] = bboxes_idx[2] - bboxes_idx_width
                bboxes_idx[1] = bboxes_idx[1] + bboxes_idx_height
                bboxes_idx[3] = bboxes_idx[3] - bboxes_idx_height

        return tracker_bboxes


    def update_bbox_info(self, img, bboxes, tracker_bboxes):

        bbox_info_list = []
        for idx in range(len(tracker_bboxes)):
            # init 
            bbox_info_dict = load_objectinfo()

            if self.demo_type == "lpr" and options.sort_type == "car":
                tracker_bbox = tracker_bboxes[idx]
                bbox_info_dict['track_id'] = tracker_bbox[-1]
                bbox_info_dict['car_info']['roi'] = tracker_bbox[0:4]
                bbox_info_list.append(bbox_info_dict)

            elif self.demo_type == "lpr" and options.sort_type == "plate":
                tracker_bbox = tracker_bboxes[idx]
                bbox_info_dict['track_id'] = tracker_bbox[-1]
                bbox_info_dict['plate_info']['roi'] = tracker_bbox[0:4]
                bbox_info_list.append(bbox_info_dict)

        return bbox_info_list


    def update_car_info( self, img, bboxes, bbox_info_list ):

        # 更新 car info
        for idx in range(len(bbox_info_list)):
            bbox_info_idx = bbox_info_list[idx]

            if self.demo_type == "lpr" and options.sort_type == "car":

                if options.car_attri_merge_bool:
                    bbox_info_idx['car_info']['attri'] = 'none'
                else:
                    car_bbox_list = []
                    for idx in range(len(options.car_attri_name_list)):
                        car_attri_name_idx = options.car_attri_name_list[idx]
                        if car_attri_name_idx in bboxes:
                            for idy in range(len(bboxes[car_attri_name_idx])):
                                car_bbox_list.append([*bboxes[car_attri_name_idx][idy], car_attri_name_idx])

                    # 求交集最大的车辆框
                    match_car_roi = match_bbox_iou(bbox_info_idx['car_info']['roi'], car_bbox_list)
                    if len(match_car_roi):
                        bbox_info_idx['car_info']['attri'] = match_car_roi[0][-1]

        return bbox_info_list


    def update_plate_info( self, img, bboxes, bbox_info_list ):

        # 更新 plate info
        for idx in range(len(bbox_info_list)):
            bbox_info_idx = bbox_info_list[idx]

            if self.demo_type == "lpr" and options.sort_type == "car":

                # license plate
                if options.license_plate_name in bboxes:
                    license_plate_roi_list = bboxes[options.license_plate_name]
                    # 求相交同时置信度最高的车牌框
                    match_license_plate_roi = match_car_license_plate(bbox_info_idx['car_info']['roi'], license_plate_roi_list)

                    if len(match_license_plate_roi):
                        Latent_plate = match_license_plate_roi[0][0:4]

                        bbox_info_idx['plate_info']['roi'] = Latent_plate

        # 更新 plate info
        for idx in range(len(bbox_info_list)):
            bbox_info_idx = bbox_info_list[idx]

            if self.demo_type == "lpr":
                
                if not len(bbox_info_idx['plate_info']['roi']):
                    continue
                
                if self.demo_type == "lpr" and options.sort_type == "plate":
    
                    # car
                    if options.car_attri_merge_name in bboxes:
                        car_roi_list = bboxes[options.car_attri_merge_name]
                        # 求相交同时置信度最高的车牌框
                        match_car_roi = match_car_list_license_plate(car_roi_list, bbox_info_idx['plate_info']['roi'])

                        if len(match_car_roi):
                            car_roi = match_car_roi[0][0:4]

                            bbox_info_idx['car_info']['roi'] = car_roi

                    for car_attri_idx in range(len(options.car_attri_name_list)):
                        car_attri_name = options.car_attri_name_list[car_attri_idx]
                        if car_attri_name in bboxes:
                            car_roi_list = bboxes[car_attri_name]
                            # 求相交同时置信度最高的车牌框
                            match_car_roi = match_car_list_license_plate(car_roi_list, bbox_info_idx['plate_info']['roi'])

                            if len(match_car_roi):
                                car_roi = match_car_roi[0][0:4]

                                bbox_info_idx['car_info']['roi'] = car_roi

                if self.country_type == "china":
                    
                    # lincense plate reader
                    # crop
                    x1 = min(max(0, int(bbox_info_idx['plate_info']['roi'][0] - options.lpr_ocr_width_expand_ratio * (bbox_info_idx['plate_info']['roi'][2] - bbox_info_idx['plate_info']['roi'][0]))), options.image_width)
                    x2 = min(max(0, int(bbox_info_idx['plate_info']['roi'][2] + options.lpr_ocr_width_expand_ratio * (bbox_info_idx['plate_info']['roi'][2] - bbox_info_idx['plate_info']['roi'][0]))), options.image_width)
                    y1 = min(max(0, int(bbox_info_idx['plate_info']['roi'][1])), options.image_height)
                    y2 = min(max(0, int(bbox_info_idx['plate_info']['roi'][3])), options.image_height)
                    crop_img = img[y1: y2, x1: x2]

                    # seg
                    seg_bbox, seg_info = self.lpr_seg.run(crop_img)

                    # ocr
                    # gray_img
                    if options.lpr_paddle_bool: 
                        plate_ocr, plate_scors_list = self.lpr.run(crop_img)
                    else:
                        gray_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                        plate_ocr, plate_scors_list = self.lpr.run(gray_crop_img)
                    
                    bbox_info_idx['plate_info']['num'] = plate_ocr
                    bbox_info_idx['plate_info']['score'] = np.array(plate_scors_list).mean()
                    bbox_info_idx['plate_info']['color'] = seg_info['color']

        return bbox_info_list


    def update_cache_container(self, img, frame_idx, bbox_info_list):
        if frame_idx % options.cache_interval == 0:
            self.params_dict['bbox_info_container'].append({'img': img, 'bbox_info': bbox_info_list})

        if len(self.params_dict['bbox_info_container']) > options.cache_container_length:
            self.params_dict['bbox_info_container'].pop(0)


    def update_bbox_state_container(self, bbox_info_list):
        
        # update
        pop_key_list = []
        for key, bbox_state_idy in self.params_dict['bbox_state_container'].items():
            # pop
            if bbox_state_idy['state']['disappear_frame_num'] > options.bbox_state_container_length:
                pop_key_list.append(key)
            bbox_state_idy['state']['disappear_frame_num'] += 1
        
        # pop
        for idx in range(len(pop_key_list)):
            self.params_dict['bbox_state_container'].pop(pop_key_list[idx])

        # 遍历单帧结果
        for idx in range(len(bbox_info_list)):
            bbox_info_idx = bbox_info_list[idx]

            is_new_id_bool = True

            # 遍历容器
            for key, bbox_state_idy in self.params_dict['bbox_state_container'].items():

                # 容器中存在追踪对象
                if bbox_info_idx['track_id'] == bbox_state_idy['track_id']:

                    is_new_id_bool = False
                    bbox_state_idy['state']['frame_num'] += 1

                    # 更新车辆速度
                    if self.demo_type == "lpr":
                        bbox_state_idy['car_info']['roi'] = bbox_info_idx['car_info']['roi'] 
                        bbox_state_idy['plate_info']['roi'] = bbox_info_idx['plate_info']['roi'] 
                        if options.sort_type == "car":
                            new_stable_loc = options.update_state_stable_loc_alpha * bbox_state_idy['state']['stable_loc'] +  (1 - options.update_state_stable_loc_alpha) * bbox_info_idx['car_info']['roi']
                        elif options.sort_type == "plate":
                            new_stable_loc = options.update_state_stable_loc_alpha * bbox_state_idy['state']['stable_loc'] +  (1 - options.update_state_stable_loc_alpha) * bbox_info_idx['plate_info']['roi']

                    old_stable_center_x = ( bbox_state_idy['state']['stable_loc'][0] + bbox_state_idy['state']['stable_loc'][2] ) / 2.0
                    new_stable_center_x = ( new_stable_loc[0] + new_stable_loc[2] ) / 2.0
                    old_stable_center_y = ( bbox_state_idy['state']['stable_loc'][1] + bbox_state_idy['state']['stable_loc'][3] ) / 2.0
                    new_stable_center_y = ( new_stable_loc[1] + new_stable_loc[3] ) / 2.0
                    bbox_state_idy['state']['up_down_speed'] = (old_stable_center_y - new_stable_center_y) / float(bbox_state_idy['state']['disappear_frame_num'])
                    bbox_state_idy['state']['left_right_speed'] = (old_stable_center_x - new_stable_center_x) / float(bbox_state_idy['state']['disappear_frame_num'])
                    bbox_state_idy['state']['stable_loc'] = new_stable_loc
                    bbox_state_idy['state']['disappear_frame_num'] = 0

                    # 车辆状态判断（上下行）
                    if bbox_state_idy['state']['up_down_speed'] > options.update_state_threshold:
                        bbox_state = 'Far'
                    elif bbox_state_idy['state']['up_down_speed'] < ( -1 * options.update_state_threshold ):
                        bbox_state = 'Near'
                    else:
                        bbox_state = "Stop"

                    if bbox_state_idy['state']['up_down_state'] != bbox_state:
                        if bbox_state_idy['state']['up_down_state_frame_num'] > 0:
                            bbox_state_idy['state']['up_down_state_frame_num'] -= 1
                        else:
                            bbox_state_idy['state']['up_down_state'] = bbox_state
                            bbox_state_idy['state']['up_down_state_frame_num'] = 0
                    else:
                        bbox_state_idy['state']['up_down_state_frame_num'] = min( bbox_state_idy['state']['up_down_state_frame_num'] + 1 , options.update_state_num_threshold)
                    
                    # 车辆状态判断（左右行）
                    if bbox_state_idy['state']['left_right_speed'] > options.update_state_threshold:
                        bbox_state = 'Left'
                    elif bbox_state_idy['state']['left_right_speed'] < ( -1 * options.update_state_threshold ):
                        bbox_state = 'Right'
                    else:
                        bbox_state = "Stop"

                    if bbox_state_idy['state']['left_right_state'] != bbox_state:
                        if bbox_state_idy['state']['left_right_state_frame_num'] > 0:
                            bbox_state_idy['state']['left_right_state_frame_num'] -= 1
                        else:
                            bbox_state_idy['state']['left_right_state'] = bbox_state
                            bbox_state_idy['state']['left_right_state_frame_num'] = 0
                    else:
                        bbox_state_idy['state']['left_right_state_frame_num'] = min( bbox_state_idy['state']['left_right_state_frame_num'] + 1 , options.update_state_num_threshold)

                    if self.demo_type == "lpr":
                        
                        bool_add_lpr = False
                        bool_roi_lpr = False

                        if not options.roi_lane_line_bool:
                            if bbox_info_idx['plate_info']['num'] != '' and \
                                bbox_state_idy['plate_info']['roi'][1] > options.ROI_Up_threshold and bbox_state_idy['plate_info']['roi'][3] < options.ROI_Down_threshold and \
                                bbox_state_idy['plate_info']['roi'][0] > options.ROI_Left_threshold and bbox_state_idy['plate_info']['roi'][2] < options.ROI_Right_threshold:
                                bool_roi_lpr = True
                        else:
                            
                            if bbox_info_idx['plate_info']['num'] != '':
                                alarm_flage, car_line, intersect_point = car_in_roi(bbox_state_idy)
                                if alarm_flage:
                                    bool_roi_lpr = True
                            
                        if bool_roi_lpr:

                            # 更新车牌识别有效帧数
                            lpr_width =  bbox_state_idy['plate_info']['roi'][2] - bbox_state_idy['plate_info']['roi'][0]
                            lpr_height =  bbox_state_idy['plate_info']['roi'][3] - bbox_state_idy['plate_info']['roi'][1]

                            if self.country_type == "china":

                                # 按照车牌宽高过滤车牌
                                if (lpr_height > options.plate_height[0]) and \
                                    (lpr_height < options.plate_height[1]) and \
                                    (lpr_width > options.plate_width[0]) and \
                                    (lpr_width < options.plate_width[1]):
                                    bool_add_lpr = True

                        if bool_add_lpr:
                            bbox_state_idy['state']['obj_num'] += 1                   
                            bbox_state_idy['state']['obj_disappear_num'] = 0
                            bbox_state_idy['state']['lpr_kind_list'].append(bbox_info_idx['plate_info']['kind'])
                            bbox_state_idy['state']['lpr_num_list'].append(bbox_info_idx['plate_info']['num'])
                            bbox_state_idy['state']['lpr_score_list'].append(bbox_info_idx['plate_info']['score'])
                            bbox_state_idy['state']['lpr_column_list'].append(bbox_info_idx['plate_info']['column'])
                            bbox_state_idy['state']['lpr_color_list'].append(bbox_info_idx['plate_info']['color'])
                            bbox_state_idy['state']['lpr_country_list'].append(bbox_info_idx['plate_info']['country'])
                            bbox_state_idy['state']['lpr_city_list'].append(bbox_info_idx['plate_info']['city'])
                            bbox_state_idy['state']['lpr_car_type_list'].append(bbox_info_idx['plate_info']['car_type'])

                            if len( bbox_state_idy['state']['lpr_num_list'] ) > options.lpr_ocr_state_container_length:
                                bbox_state_idy['state']['lpr_num_list'].pop(0)

                            if len( bbox_state_idy['state']['lpr_kind_list'] ) > options.lpr_ocr_state_container_length:
                                bbox_state_idy['state']['lpr_kind_list'].pop(0)

                            if len( bbox_state_idy['state']['lpr_score_list'] ) > options.lpr_ocr_state_container_length:
                                bbox_state_idy['state']['lpr_score_list'].pop(0)

                            if len( bbox_state_idy['state']['lpr_column_list'] ) > options.lpr_city_state_container_length:
                                bbox_state_idy['state']['lpr_column_list'].pop(0)

                            if len( bbox_state_idy['state']['lpr_color_list'] ) > options.lpr_city_state_container_length:
                                bbox_state_idy['state']['lpr_color_list'].pop(0)

                            if len( bbox_state_idy['state']['lpr_country_list'] ) > options.lpr_city_state_container_length:
                                bbox_state_idy['state']['lpr_country_list'].pop(0)

                            if len( bbox_state_idy['state']['lpr_city_list'] ) > options.lpr_city_state_container_length:
                                bbox_state_idy['state']['lpr_city_list'].pop(0)

                            if len( bbox_state_idy['state']['lpr_car_type_list'] ) > options.lpr_city_state_container_length:
                                bbox_state_idy['state']['lpr_car_type_list'].pop(0)

                        else:
                            bbox_state_idy['state']['obj_disappear_num'] += 1

                    # 信息同步
                    bbox_info_idx['state']['frame_num'] = bbox_state_idy['state']['frame_num']
                    bbox_info_idx['state']['stable_loc'] = bbox_state_idy['state']['stable_loc']
                    bbox_info_idx['state']['up_down_speed'] = bbox_state_idy['state']['up_down_speed']
                    bbox_info_idx['state']['left_right_speed'] = bbox_state_idy['state']['left_right_speed']
                    bbox_info_idx['state']['up_down_state'] = bbox_state_idy['state']['up_down_state']
                    bbox_info_idx['state']['up_down_state_frame_num'] = bbox_state_idy['state']['up_down_state_frame_num']
                    bbox_info_idx['state']['left_right_state'] = bbox_state_idy['state']['left_right_state']
                    bbox_info_idx['state']['left_right_state_frame_num'] = bbox_state_idy['state']['left_right_state_frame_num']
                    bbox_info_idx['state']['obj_num'] = bbox_state_idy['state']['obj_num']                 
                    bbox_info_idx['state']['obj_disappear_num'] = bbox_state_idy['state']['obj_disappear_num']                 

            if is_new_id_bool:
                
                # init
                bbox_state_idy = load_objectinfo()

                bbox_state_idy['track_id'] = bbox_info_idx['track_id']
                bbox_state_idy['state']['frame_num'] += 1               
                
                if self.demo_type == "lpr":
                    bbox_state_idy['car_info']['roi'] = bbox_info_idx['car_info']['roi']
                    bbox_state_idy['plate_info']['roi'] = bbox_info_idx['plate_info']['roi']
                    if options.sort_type == "car":
                        bbox_state_idy['state']['stable_loc'] = bbox_info_idx['car_info']['roi']
                    elif options.sort_type == "plate":
                        bbox_state_idy['state']['stable_loc'] = bbox_info_idx['plate_info']['roi']

                    bool_add_lpr = False
                    bool_roi_lpr = False

                    if not options.roi_lane_line_bool:
                        if bbox_info_idx['plate_info']['num'] != '' and \
                            bbox_state_idy['plate_info']['roi'][1] > options.ROI_Up_threshold and bbox_state_idy['plate_info']['roi'][3] < options.ROI_Down_threshold and \
                            bbox_state_idy['plate_info']['roi'][0] > options.ROI_Left_threshold and bbox_state_idy['plate_info']['roi'][2] < options.ROI_Right_threshold:
                            bool_roi_lpr = True
                    else:
                        if bbox_info_idx['plate_info']['num'] != '':
                            alarm_flage, car_line, intersect_point = car_in_roi(bbox_state_idy)
                            if alarm_flage:
                                bool_roi_lpr = True
                        
                    if bool_roi_lpr:
                        # 更新车牌识别有效帧数
                        lpr_width =  bbox_state_idy['plate_info']['roi'][2] - bbox_state_idy['plate_info']['roi'][0]
                        lpr_height =  bbox_state_idy['plate_info']['roi'][3] - bbox_state_idy['plate_info']['roi'][1]

                        if self.country_type == "china":

                            # 按照车牌宽高过滤车牌
                            if (lpr_height > options.plate_height[0]) and \
                                (lpr_height < options.plate_height[1]) and \
                                (lpr_width > options.plate_width[0]) and \
                                (lpr_width < options.plate_width[1]):
                                bool_add_lpr = True
                
                    if bool_add_lpr:
                        bbox_state_idy['state']['obj_num'] += 1
                        bbox_state_idy['state']['obj_disappear_num'] = 0
                        bbox_state_idy['state']['lpr_kind_list'].append(bbox_info_idx['plate_info']['kind'])
                        bbox_state_idy['state']['lpr_num_list'].append(bbox_info_idx['plate_info']['num'])
                        bbox_state_idy['state']['lpr_score_list'].append(bbox_info_idx['plate_info']['score'])
                        bbox_state_idy['state']['lpr_column_list'].append(bbox_info_idx['plate_info']['column'])
                        bbox_state_idy['state']['lpr_color_list'].append(bbox_info_idx['plate_info']['color'])
                        bbox_state_idy['state']['lpr_country_list'].append(bbox_info_idx['plate_info']['country'])
                        bbox_state_idy['state']['lpr_city_list'].append(bbox_info_idx['plate_info']['city'])
                        bbox_state_idy['state']['lpr_car_type_list'].append(bbox_info_idx['plate_info']['car_type'])
                        
                    else:
                        bbox_state_idy['state']['obj_disappear_num'] += 1
                    
                # 信息同步
                bbox_info_idx['state']['frame_num'] = bbox_state_idy['state']['frame_num']
                bbox_info_idx['state']['stable_loc'] = bbox_state_idy['state']['stable_loc']
                bbox_info_idx['state']['obj_num'] = bbox_state_idy['state']['obj_num']   
                bbox_info_idx['state']['obj_disappear_num'] = bbox_state_idy['state']['obj_disappear_num']   
                
                self.params_dict['bbox_state_container'][bbox_state_idy['track_id']] = bbox_state_idy

        # update
        # center_point_list
        pop_key_list = []
        for key, bbox_state_idy in self.params_dict['bbox_state_container'].items():

            bbox_state_idy['state']['center_point_list'].append( ( (bbox_state_idy['state']['stable_loc'][0] + bbox_state_idy['state']['stable_loc'][2]) / 2 , (bbox_state_idy['state']['stable_loc'][1] + bbox_state_idy['state']['stable_loc'][3]) / 2 ) )
            
            if len( bbox_state_idy['state']['center_point_list'] ) > options.bbox_state_container_length:
                bbox_state_idy['state']['center_point_list'].pop(0)

        return bbox_info_list


    def update_capture_dict(self):
        
        # update
        pop_key_list = []
        for key, capture_dict_idy in self.params_dict['capture_container'].items():
          
            # pop
            if capture_dict_idy['capture']['capture_frame_num'] > options.capture_frame_num_threshold:
                pop_key_list.append(key)
            
            elif capture_dict_idy['capture']['capture_bool']:
                pop_key_list.append(key)
            
            capture_dict_idy['capture']['capture_frame_num'] += 1
        
        # pop
        for idx in range(len(pop_key_list)):
            self.params_dict['capture_container'].pop(pop_key_list[idx])
        
        # 报警逻辑
        # 1、快速抓怕：车辆行驶到触发线，执行抓拍。（近处、远处、左边、右边）
        # 2、稳定抓拍：车牌位于图像正中央，超过一定时间，执行抓拍。
        for _, bbox_state_idy in self.params_dict['bbox_state_container'].items():
            # init 
            near_flage = False
            far_flage = False
            left_flage = False
            right_flage = False
            outtime_flage_01 = False
            outtime_flage_02 = False
            report_flage = False
            outtime_flage_double_01 = False

            if bbox_state_idy['state']['obj_disappear_num'] == 0:
                
                if self.demo_type == "lpr":
                    loc_center_x = (bbox_state_idy['plate_info']['roi'][0] + bbox_state_idy['plate_info']['roi'][2]) / 2.0
                    loc_center_y = (bbox_state_idy['plate_info']['roi'][1] + bbox_state_idy['plate_info']['roi'][3]) / 2.0

                # 如果车辆向近处行驶, bbox_state_idy['state']['up_down_state_frame_num'] >= 3 条件用于避免刚进 ROI 或者车辆静止状态下的误判
                if bbox_state_idy['state']['up_down_state'] == 'Near' and bbox_state_idy['state']['up_down_state_frame_num'] >= 3:
                    if abs(loc_center_y - options.Down_threshold) < options.capture_up_down_distance_boundary_threshold and \
                        bbox_state_idy['state']['obj_num'] > options.capture_info_frame_threshold:
                        near_flage = True

                # 如果车辆向远处行驶，bbox_state_idy['state']['up_down_state_frame_num'] 条件用于避免刚进 ROI 或者车辆静止状态下的误判
                if bbox_state_idy['state']['up_down_state'] == 'Far' and bbox_state_idy['state']['up_down_state_frame_num'] >= 3:
                    if abs(loc_center_y - options.Up_threshold) < options.capture_up_down_distance_boundary_threshold and \
                        bbox_state_idy['state']['obj_num'] > options.capture_info_frame_threshold:
                        far_flage = True

                # 如果车辆向左边行驶
                if bbox_state_idy['state']['left_right_state'] == 'Left' and bbox_state_idy['state']['left_right_state_frame_num'] >= 3:
                    if not options.roi_lane_line_bool:
                        if (( loc_center_x - options.Left_threshold > 0 and \
                            loc_center_x - options.Left_threshold < options.capture_left_right_distance_near_boundary_threshold ) or \
                            ( options.Left_threshold - loc_center_x > 0 and \
                            options.Left_threshold - loc_center_x < options.capture_left_right_distance_far_boundary_threshold )) and \
                            bbox_state_idy['state']['obj_num'] > options.capture_info_frame_threshold:
                            left_flage = True
                    else:
                        dist_left_lane_line = ((loc_center_y - options.Left_line_b) / (options.Left_line_k + 1e-5) - loc_center_x)
                        dist_left_lane_line = (-1) * dist_left_lane_line if (options.Left_line_k < 0) else dist_left_lane_line
                        if (( dist_left_lane_line > 0 and \
                            dist_left_lane_line < options.capture_left_right_distance_near_boundary_threshold ) or \
                            ( dist_left_lane_line < 0 and \
                            (-1) * dist_left_lane_line < options.capture_left_right_distance_far_boundary_threshold )) and \
                            bbox_state_idy['state']['obj_num'] > options.capture_info_frame_threshold:
                            left_flage = True

                # 如果车辆向右边行驶
                if bbox_state_idy['state']['left_right_state'] == 'Right' and bbox_state_idy['state']['left_right_state_frame_num'] >= 3:
                    if not options.roi_lane_line_bool:
                        if (( loc_center_x - options.Right_threshold > 0 and \
                            loc_center_x - options.Right_threshold < options.capture_left_right_distance_far_boundary_threshold ) or \
                            ( options.Right_threshold - loc_center_x > 0 and \
                            options.Right_threshold - loc_center_x < options.capture_left_right_distance_near_boundary_threshold )) and \
                            bbox_state_idy['state']['obj_num'] > options.capture_info_frame_threshold:
                            right_flage = True
                    else:
                        dist_right_lane_line = ((loc_center_y - options.Right_line_b) / (options.Right_line_k + 1e-5) - loc_center_x)
                        dist_right_lane_line = (-1) * dist_right_lane_line if (options.Right_line_k > 0) else dist_right_lane_line
                        if (( dist_right_lane_line > 0 and \
                            dist_right_lane_line < options.capture_left_right_distance_far_boundary_threshold ) or \
                            ( dist_right_lane_line < 0 and \
                            (-1) * dist_right_lane_line < options.capture_left_right_distance_near_boundary_threshold )) and \
                            bbox_state_idy['state']['obj_num'] > options.capture_info_frame_threshold:
                            right_flage = True

                # 如果车辆在视野内，超过 25 帧
                if bbox_state_idy['state']['obj_num'] > options.capture_outtime_frame_threshold_01:
                    outtime_flage_01 = True

                # 如果车辆在视野内，超过 150 帧
                if bbox_state_idy['state']['obj_num'] > options.capture_outtime_frame_threshold_02:
                    outtime_flage_02 = True

            # 更新 capture_dict 抓拍字典
            capture_dict = load_objectinfo()
            capture_dict['track_id'] = bbox_state_idy['track_id']

            if near_flage and not bbox_state_idy['capture']['near_report_flage']:
                bbox_state_idy['capture']['near_report_flage'] = True
                report_flage = True
                capture_dict['capture']['flage'] = 'near_flage'

            if far_flage and not bbox_state_idy['capture']['far_report_flage']:
                bbox_state_idy['capture']['far_report_flage'] = True
                report_flage = True
                capture_dict['capture']['flage'] = 'far_flage'
                
            if left_flage and not bbox_state_idy['capture']['left_report_flage']:
                bbox_state_idy['capture']['left_report_flage'] = True
                report_flage = True
                capture_dict['capture']['flage'] = 'left_flage'

            if right_flage and not bbox_state_idy['capture']['right_report_flage']:
                bbox_state_idy['capture']['right_report_flage'] = True
                report_flage = True
                capture_dict['capture']['flage'] = 'right_flage'

            if outtime_flage_01 and not bbox_state_idy['capture']['outtime_flage_01']:
                bbox_state_idy['capture']['outtime_flage_01'] = True
                report_flage = True
                capture_dict['capture']['flage'] = 'outtime_flage_01'

            if outtime_flage_02 and not bbox_state_idy['capture']['outtime_flage_02']:
                bbox_state_idy['capture']['outtime_flage_02'] = True
                report_flage = True
                capture_dict['capture']['flage'] = 'outtime_flage_02'

            if outtime_flage_double_01 and not bbox_state_idy['capture']['outtime_flage_double_01']:
                bbox_state_idy['capture']['outtime_flage_double_01'] = True
                report_flage = True
                capture_dict['capture']['flage'] = 'outtime_flage_double_01'

            if report_flage:
                
                # 更新 capture_dict 抓拍字典
                if bbox_state_idy['track_id'] not in self.params_dict['capture_container']:

                    self.params_dict['capture_container'][capture_dict['track_id']] = capture_dict
            
        return


    def update_capture_state(self):
        
        # update
        pop_key_list = []
        for key, capture_res_dict_idy in self.params_dict['capture_res_container'].items():
          
            # pop
            if capture_res_dict_idy['capture']['capture_frame_num'] > options.capture_clear_frame_num_threshold:
                pop_key_list.append(key)
            
            capture_res_dict_idy['capture']['capture_frame_num'] += 1
        
        # pop
        for idx in range(len(pop_key_list)):
            self.params_dict['capture_res_container'].pop(pop_key_list[idx])

        # 抓拍逻辑
        # 1、查找稳定结果
        for capture_id_idx, capture_dict_idy in self.params_dict['capture_container'].items():
            
            # init 
            capture_res_dict = load_objectinfo()

            for _, bbox_state_idy in self.params_dict['bbox_state_container'].items():

                if bbox_state_idy['track_id'] == capture_id_idx:
                    
                    if self.demo_type == "lpr":
                        
                        lpr_kind_np = np.array(bbox_state_idy['state']['lpr_kind_list'])
                        lpr_num_np = np.array(bbox_state_idy['state']['lpr_num_list'])
                        lpr_score_np = np.array(bbox_state_idy['state']['lpr_score_list'])
                        lpr_color_np = np.array(bbox_state_idy['state']['lpr_color_list'])
                        lpr_country_np = np.array(bbox_state_idy['state']['lpr_country_list'])
                        lpr_city_np = np.array(bbox_state_idy['state']['lpr_city_list'])
                        lpr_car_type_np = np.array(bbox_state_idy['state']['lpr_car_type_list'])
                        lpr_column_np = np.array(bbox_state_idy['state']['lpr_column_list'])
                        lpr_color_np = lpr_color_np[lpr_color_np != "none"]

                        # 获得抓拍序列
                        if len(lpr_num_np[lpr_score_np > options.capture_lpr_score_threshold]):
                            
                            if len(list(lpr_kind_np)):
                                capture_lpr_kind, capture_lpr_kind_frame = Counter(list(lpr_kind_np[lpr_score_np > options.capture_lpr_score_threshold])).most_common(1)[0]
                            if len(list(lpr_num_np)):
                                capture_lpr_num, capture_lpr_num_frame = Counter(list(lpr_num_np[lpr_score_np > options.capture_lpr_score_threshold])).most_common(1)[0]
                            if len(list(lpr_color_np)):
                                capture_lpr_color, capture_lpr_color_frame = Counter(list(lpr_color_np)).most_common(1)[0]
                            if len(list(lpr_country_np)):
                                capture_lpr_country, capture_lpr_country_frame = Counter(list(lpr_country_np)).most_common(1)[0]
                            if len(list(lpr_city_np)):
                                capture_lpr_city, capture_lpr_city_frame = Counter(list(lpr_city_np)).most_common(1)[0]
                            if len(list(lpr_car_type_np)):
                                capture_lpr_car_type, capture_lpr_car_type_frame = Counter(list(lpr_car_type_np)).most_common(1)[0]
                            if len(list(lpr_column_np)):
                                capture_lpr_column, capture_lpr_column_frame = Counter(list(lpr_column_np)).most_common(1)[0]

                            capture_from_container_list = self.find_capture_plate(bbox_state_idy['track_id'], capture_lpr_num)

                            if self.country_type == "china":
                                if capture_lpr_num_frame >= options.capture_lpr_num_frame_threshold and \
                                    len(capture_from_container_list) and \
                                    capture_lpr_num not in self.params_dict['capture_res_container']:

                                    # 状态容器更新
                                    bbox_state_idy['capture_occupying']['cap_bool'] = True
                                    bbox_state_idy['capture_occupying']['cap_lpr_num'] = capture_lpr_num
                                    bbox_state_idy['capture_occupying']['cap_lpr_color'] = "none"
                                    if len(list(lpr_color_np)) and capture_lpr_color_frame >= options.capture_lpr_color_frame_threshold:
                                        bbox_state_idy['capture_occupying']['cap_lpr_color'] = capture_lpr_color

                                    capture_res_dict['track_id'] = capture_id_idx
                                    capture_res_dict['capture']['flage'] = capture_dict_idy['capture']['flage']
                                    capture_res_dict['capture']['img_bbox_info_list'] = capture_from_container_list
                                    capture_res_dict['capture']['capture_bool'] = True
                                    capture_res_dict['plate_info']['num'] = capture_lpr_num
                                    
                                    # color
                                    capture_res_dict['plate_info']['color'] = "none"
                                    if len(list(lpr_color_np)) and capture_lpr_color_frame >= options.capture_lpr_color_frame_threshold:
                                        capture_res_dict['plate_info']['color'] = capture_lpr_color
                                               
                                    self.params_dict['capture_res_container'][capture_res_dict['plate_info']['num']] = capture_res_dict

                                    # 信息同步 
                                    capture_dict_idy['capture']['capture_bool'] = capture_res_dict['capture']['capture_bool']
                                    capture_dict_idy['plate_info']['num'] = capture_res_dict['plate_info']['num']
                                    capture_dict_idy['plate_info']['color'] = capture_res_dict['plate_info']['color']

        return


    def find_capture_plate(self, captute_id, capture_lpr_num):
        
        capture_from_container_list = []

        for idy in range(len(self.params_dict['bbox_info_container'])):
            bbox_info_list = self.params_dict['bbox_info_container'][idy]['bbox_info']

            for idx in range(len(bbox_info_list)):
                bbox_info_idx = bbox_info_list[idx]

                # 容器中存在追踪对象
                if bbox_info_idx['track_id'] == captute_id and bbox_info_idx['plate_info']['num'] == capture_lpr_num:
                    capture_from_container_list.append(self.params_dict['bbox_info_container'][idy])
        
        if len(capture_from_container_list) > 3:
            capture_from_container_list = random.sample(capture_from_container_list, 3)
        
        return capture_from_container_list
    

    def update_bbox_state_container_capture_occupying(self, img, bbox_info_list):

        # update
        pop_key_list = []
        for key, capture_dict_idy in self.params_dict['capture_occupying_container'].items():
          
            # pop
            if capture_dict_idy['state_occupying']['disappear_frame_num'] > options.capture_frame_num_threshold:
                pop_key_list.append(key)
            
            elif capture_dict_idy['capture_occupying']['capture_bool']:
                pop_key_list.append(key)
            
            capture_dict_idy['state_occupying']['disappear_frame_num'] += 1
        
        # pop
        for idx in range(len(pop_key_list)):
            self.params_dict['capture_occupying_container'].pop(pop_key_list[idx])

        # update
        pop_key_list = []
        for key, capture_res_dict_idy in self.params_dict['capture_occupying_res_container'].items():
          
            # pop
            if capture_res_dict_idy['capture']['capture_frame_num'] > options.capture_clear_frame_num_threshold:
                pop_key_list.append(key)
            
            capture_res_dict_idy['capture']['capture_frame_num'] += 1
        
        # pop
        for idx in range(len(pop_key_list)):
            self.params_dict['capture_occupying_res_container'].pop(pop_key_list[idx])
        
        # 遍历单帧结果
        for idx in range(len(bbox_info_list)):
            bbox_info_idx = bbox_info_list[idx]

            # 遍历容器
            for key, bbox_state_idy in self.params_dict['bbox_state_container'].items():

                # 容器中存在追踪对象
                if bbox_info_idx['track_id'] == bbox_state_idy['track_id']:
                                        
                    # 1、判断车辆是否位于占道区域内部
                    alarm_flage, car_line, intersect_point = car_in_roi(bbox_state_idy)
                    
                    # 信息同步
                    bbox_info_idx['state_occupying']['car_line'] = car_line
                    bbox_info_idx['state_occupying']['intersect_point'] = intersect_point
                    bbox_info_idx['state_occupying']['frame_num'] = -1
                    bbox_info_idx['state_occupying']['disappear_frame_num'] = 0

                    if bbox_state_idy['track_id'] not in self.params_dict['capture_occupying_container']:
                        
                        capture_occupying = load_objectinfo()

                        # 保存第一张图片，需要确保图像中有车牌
                        if alarm_flage and len(bbox_state_idy['plate_info']['roi']):

                            # 更新 capture_occupying 抓拍字典
                            capture_occupying['track_id'] = bbox_state_idy['track_id']
                            capture_occupying['state_occupying']['frame_num'] = 1
                            capture_occupying['state_occupying']['disappear_frame_num'] = 0

                            # capture_occupying['capture_occupying']['cap_lpr_num'] = ''
                            # capture_occupying['capture_occupying']['cap_lpr_color'] = ''
                            capture_occupying['capture_occupying']['cap_img'].append(img)
                            capture_occupying['capture_occupying']['cap_plate_roi'].append(bbox_state_idy['plate_info']['roi'])
                            # objectinfo['capture_occupying']['capture_bool'] = False                  # 抓拍成功标志

                            self.params_dict['capture_occupying_container'][bbox_state_idy['track_id']] = capture_occupying

                        # 信息同步
                        bbox_info_idx['state_occupying']['frame_num'] = capture_occupying['state_occupying']['frame_num']
                        bbox_info_idx['state_occupying']['disappear_frame_num'] = capture_occupying['state_occupying']['disappear_frame_num']

                    else:
                        
                        capture_occupying = self.params_dict['capture_occupying_container'][bbox_state_idy['track_id']]

                        if alarm_flage:          

                            capture_occupying['state_occupying']['frame_num'] += 1
                            capture_occupying['state_occupying']['disappear_frame_num'] = 0

                            # 保存第二张图片
                            if capture_occupying['state_occupying']['frame_num'] == int(options.capture_occupying_frame_threshold / 2):
                                capture_occupying['capture_occupying']['cap_img'].append(img)
                                capture_occupying['capture_occupying']['cap_plate_roi'].append(bbox_state_idy['plate_info']['roi'])
                            
                            # 保存第三张图片
                            elif capture_occupying['state_occupying']['frame_num'] == (options.capture_occupying_frame_threshold):
                                capture_occupying['capture_occupying']['cap_img'].append(img)
                                capture_occupying['capture_occupying']['cap_plate_roi'].append(bbox_state_idy['plate_info']['roi'])
                            
                            # 判断是否有稳定车牌抓拍（若超时仍无稳定车牌抓拍，归零后重置抓拍）
                            elif capture_occupying['state_occupying']['frame_num'] >= (options.capture_occupying_frame_threshold) and \
                                capture_occupying['state_occupying']['frame_num'] < options.capture_occupying_outtime_frame_threshold:

                                # 信息同步
                                capture_occupying['capture_occupying']['cap_bool'] = bbox_state_idy['capture_occupying']['cap_bool']
                                capture_occupying['capture_occupying']['cap_lpr_num'] = bbox_state_idy['capture_occupying']['cap_lpr_num']
                                capture_occupying['capture_occupying']['cap_lpr_color'] = bbox_state_idy['capture_occupying']['cap_lpr_color']

                                if capture_occupying['capture_occupying']['cap_lpr_num'] not in self.params_dict['capture_occupying_res_container'] and \
                                    capture_occupying['capture_occupying']['cap_bool']:
        
                                        capture_occupying['capture_occupying']['capture_bool'] = True

                                        # init 
                                        capture_res_dict = load_objectinfo()

                                        capture_res_dict['track_id'] = capture_occupying['track_id']
                                        capture_res_dict['state_occupying']['frame_num'] = capture_occupying['state_occupying']['frame_num']
                                        capture_res_dict['state_occupying']['disappear_frame_num'] = capture_occupying['state_occupying']['disappear_frame_num']
                                        
                                        capture_res_dict['capture_occupying']['cap_lpr_num'] = capture_occupying['capture_occupying']['cap_lpr_num']
                                        capture_res_dict['capture_occupying']['cap_lpr_color'] = capture_occupying['capture_occupying']['cap_lpr_color']
                                        capture_res_dict['capture_occupying']['cap_img'] = capture_occupying['capture_occupying']['cap_img']
                                        capture_res_dict['capture_occupying']['cap_plate_roi'] = capture_occupying['capture_occupying']['cap_plate_roi']
                                        capture_res_dict['capture_occupying']['capture_bool'] = capture_occupying['capture_occupying']['capture_bool']
                                        self.params_dict['capture_occupying_res_container'][capture_occupying['capture_occupying']['cap_lpr_num']] = capture_res_dict
                            elif capture_occupying['state_occupying']['frame_num'] >= options.capture_occupying_outtime_frame_threshold:
                                
                                # 若超时仍无稳定车牌抓拍，归零后重置抓拍
                                capture_occupying['state_occupying']['disappear_frame_num'] = options.capture_frame_num_threshold + 1

                        # 信息同步
                        bbox_info_idx['state_occupying']['frame_num'] = capture_occupying['state_occupying']['frame_num']
                        bbox_info_idx['state_occupying']['disappear_frame_num'] = capture_occupying['state_occupying']['disappear_frame_num']