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
from Image.detection2d.mmdetection.demo.detector.yolov6_landmark_degree_detector import YOLOV6LandmarkDegreeDetector
from Image.detection2d.mmdetection.demo.detector.yolov6_landmark_detector import YOLOV6LandmarkDetector
from Image.recognition2d.lpr.infer.lpr import LPRCaffe, LPRPytorch
from Image.recognition2d.lpr.infer.lpr_seg import LPRSegColorClassCaffe
from Image.Demo.street_sweeping_capture.sort.mot_sort import Sort

from Image.Demo.street_sweeping_capture.info.options import *
from Image.Demo.street_sweeping_capture.info.param import *
from Image.Demo.street_sweeping_capture.utils.iou import *
from Image.Demo.street_sweeping_capture.utils.landmark2degree import landmark2degree

import Image.recognition2d.script.paddle.infer.lpr as paddle_lpr


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


    def model_init(self):

        if self.demo_type == "lpr":
            # detector
            if options.ssd_bool:
                self.detector = SSDDetector(prototxt=options.lpr.ssd_prototxt, model_path=options.lpr.ssd_model_path, ssd_caffe_bool=options.ssd_caffe_bool, ssd_openvino_bool=options.ssd_openvino_bool, merge_class_bool=options.lpr.car_attri_merge_bool, gpu_bool=options.gpu_bool)
            elif options.yolov6_bool:
                self.detector = YOLOV6Detector(options.lpr.yolov6_config, options.lpr.yolov6_checkpoint, class_name=options.lpr.yolov6_class_name, threshold_list=options.lpr.yolov6_threshold_list, device=options.device)

            # lincense plate reader
            if self.country_type == "china":
                if options.lpr.lpr_caffe_bool:
                    if lpr.lpr_paddle_bool: 
                        self.lpr = paddle_lpr.LPRCaffe(options.lpr.china.ocr_caffe_model_path, options.lpr.china.ocr_caffe_prototxt, options.lpr.china.ocr_labels_dict_path, img_shape=options.lpr.china.input_shape, gpu_bool=False)
                    else:
                        self.lpr = LPRCaffe(options.lpr.china.ocr_caffe_prototxt, options.lpr.china.ocr_caffe_model_path, input_shape=options.lpr.china.input_shape, ocr_labels=options.lpr.china.ocr_labels, padding_bool=options.lpr.china.padding_bool, prefix_beam_search_bool=options.lpr.china.ocr_prefix_beam_search_bool, gpu_bool=False)
                    self.lpr_seg = LPRSegColorClassCaffe(options.lpr.china.seg_caffe_prototxt, options.lpr.china.seg_caffe_model_path, options.lpr.china.seg_city_dict_name, options.lpr.china.seg_color_dict_name, input_shape=options.lpr.china.seg_input_shape, city_bool=options.lpr.china.seg_city_bool, color_bool=options.lpr.china.seg_color_bool, gpu_bool=False)

                elif options.lpr.lpr_pytorch_bool:
                    if lpr.lpr_paddle_bool: 
                        self.lpr = None
                    else:
                        self.lpr = LPRPytorch(options.lpr.china.ocr_pth_path, input_shape=options.lpr.china.input_shape, ocr_labels=options.lpr.china.ocr_labels, prefix_beam_search_bool=options.lpr.china.ocr_prefix_beam_search_bool)
                    self.lpr_seg = None

            # tracker
            if options.lpr.sort_type == "car":
                self.mot_tracker = Sort(max_age=options.lpr.max_age, min_hits=options.lpr.min_hits, iou_threshold=options.lpr.iou_threshold)

        elif self.demo_type == "face":
            # detector
            if options.ssd_bool:
                self.detector = SSDDetector(prototxt=options.face.ssd_prototxt, model_path=options.face.ssd_model_path, ssd_caffe_bool=options.ssd_caffe_bool, gpu_bool=options.gpu_bool)
            elif options.yolov6_bool:
                # self.detector = YOLOV6Detector(options.face.yolov6_config, options.face.yolov6_checkpoint, class_name=options.face.yolov6_class_name, threshold_list=options.face.yolov6_threshold_list, device=options.device)
                if face.landmark_bool:
                    self.detector = YOLOV6LandmarkDetector(options.face.yolov6_config, options.face.yolov6_checkpoint, class_name=options.face.yolov6_class_name, threshold_list=options.face.yolov6_threshold_list, device=options.device)
                elif face.landmark_degree_bool:
                    self.detector = YOLOV6LandmarkDegreeDetector(options.face.yolov6_config, options.face.yolov6_checkpoint, class_name=options.face.yolov6_class_name, threshold_list=options.face.yolov6_threshold_list, device=options.device)                    
            
            # tracker
            self.mot_tracker = Sort(max_age=options.face.max_age, min_hits=options.face.min_hits, iou_threshold=options.face.iou_threshold)


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

        # update face info
        bbox_info_list = self.update_face_info( img, bboxes, bbox_info_list )

        # store
        # 跳帧存储原图和检测识别结果
        self.update_cache_container( img, frame_idx, bbox_info_list )

        # 更新状态容器，同时更新车辆行驶状态和帧率
        bbox_info_list = self.update_bbox_state_container( bbox_info_list )
        bbox_state_container = self.params_dict['bbox_state_container']

        self.update_capture_dict()
        capture_container = self.params_dict['capture_container']
        self.update_capture_state()
        capture_res_container = self.params_dict['capture_res_container']

        ## capture_line
        if options.roi_bool:
            capture_line_up_down = [ options.roi_area[1] + ( options.roi_area[3] - options.roi_area[1] ) * ratio for ratio in options.capture_line_up_down_ratio ]
            capture_line_left_right = [ options.roi_area[0] + ( options.roi_area[2] - options.roi_area[0] ) * ratio for ratio in options.capture_line_left_right_ratio ]
        else:
            capture_line_up_down = [ options.image_height * ratio for ratio in options.capture_line_up_down_ratio ]
            capture_line_left_right = [ options.image_width * ratio for ratio in options.capture_line_left_right_ratio ]

        return tracker_bboxes, bbox_info_list, bbox_state_container, capture_line_up_down, capture_line_left_right, capture_container, capture_res_container
    

    def update_tracker_bboxes(self, bboxes):

        if self.demo_type == "lpr":
            if options.lpr.sort_type == "car":
                if options.lpr.car_attri_merge_bool:
                    # tracker
                    if options.lpr.car_attri_merge_name in bboxes:
                        dets = np.array(bboxes[options.lpr.car_attri_merge_name])
                    else:
                        dets = np.empty((0, 5))
                    tracker_bboxes = self.mot_tracker.update(dets)

                else:
                    # tracker
                    dets = np.empty((0, 5))
                    for idx in range(len(options.lpr.car_attri_name_list)):
                        car_attri_name_idx = options.lpr.car_attri_name_list[idx]
                        if car_attri_name_idx in bboxes:
                            dets = np.concatenate((dets, np.array(bboxes[car_attri_name_idx])), axis=0)
                            
                    tracker_bboxes = self.mot_tracker.update(dets)

        elif self.demo_type == "face":
            # tracker
            dets = np.empty((0, 5))
            for idx in range(len(options.face.sort_class_name)):
                sort_class_name = options.face.sort_class_name[idx]
                if sort_class_name in bboxes:

                    # face
                    # 由于车牌面积太小，跟踪不上，需要增大面积
                    bboxes_class_list = []
                    
                    for idy in range(len(bboxes[sort_class_name])):
                        bboxes_idx = bboxes[sort_class_name][idy]
                        
                        bboxes_idx_width = ( bboxes_idx[2] - bboxes_idx[0] ) * options.face.sort_expand_ratio
                        bboxes_idx_height = ( bboxes_idx[3] - bboxes_idx[1] ) * options.face.sort_expand_ratio
                        bboxes_idx[0] = bboxes_idx[0] - bboxes_idx_width
                        bboxes_idx[2] = bboxes_idx[2] + bboxes_idx_width
                        bboxes_idx[1] = bboxes_idx[1] - bboxes_idx_height
                        bboxes_idx[3] = bboxes_idx[3] + bboxes_idx_height
                        bboxes_class_list.append([bboxes_idx[0], bboxes_idx[1], bboxes_idx[2], bboxes_idx[3], bboxes_idx[-1]])

                    dets = np.concatenate((dets, np.array(bboxes_class_list)), axis=0)
            
            tracker_bboxes = self.mot_tracker.update(dets)

            # face
            # 由于增加了面积，需要减少面积，获得精确坐标
            for idx in range(len(tracker_bboxes)):
                bboxes_idx = tracker_bboxes[idx]

                bboxes_idx_width = (( bboxes_idx[2] - bboxes_idx[0] ) * options.face.sort_expand_ratio ) / ( 1.0 + options.face.sort_expand_ratio * 2.0 )
                bboxes_idx_height = (( bboxes_idx[3] - bboxes_idx[1] ) * options.face.sort_expand_ratio )  / ( 1.0 + options.face.sort_expand_ratio * 2.0 )
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

            if self.demo_type == "lpr":
                if options.lpr.sort_type == "car":
                    tracker_bbox = tracker_bboxes[idx]
                    bbox_info_dict['track_id'] = tracker_bbox[-1]
                    bbox_info_dict['car_info']['roi'] = tracker_bbox[0:4]
                    bbox_info_list.append(bbox_info_dict)

            elif self.demo_type == "face":
                bbox_info_dict['track_id'] = tracker_bboxes[idx][-1]
                bbox_info_dict['face_info']['roi'] = tracker_bboxes[idx][0:4]
                bbox_info_list.append(bbox_info_dict)

        return bbox_info_list


    def update_car_info( self, img, bboxes, bbox_info_list ):

        # 遍历单帧结果
        for idx in range(len(bbox_info_list)):
            bbox_info_idx = bbox_info_list[idx]

            if self.demo_type == "lpr":
                # 车辆属性更新
                if options.lpr.car_attri_merge_bool:
                    bbox_info_idx['car_info']['attri'] = 'none'
                else:
                    car_bbox_list = []
                    for idx in range(len(options.lpr.car_attri_name_list)):
                        car_attri_name_idx = options.lpr.car_attri_name_list[idx]
                        if car_attri_name_idx in bboxes:
                            for idy in range(len(bboxes[car_attri_name_idx])):
                                car_bbox_list.append([*bboxes[car_attri_name_idx][idy], car_attri_name_idx])

                    # 求交集最大的车辆框
                    match_car_roi = match_bbox_iou(bbox_info_idx['car_info']['roi'], car_bbox_list)
                    if len(match_car_roi):
                        bbox_info_idx['car_info']['attri'] = match_car_roi[0][-1]
        
        return bbox_info_list


    def update_plate_info( self, img, bboxes, bbox_info_list ):

        # 遍历单帧结果
        for idx in range(len(bbox_info_list)):
            bbox_info_idx = bbox_info_list[idx]

            if self.demo_type == "lpr":
                # license plate
                if options.lpr.license_plate_name in bboxes:
                    license_plate_roi_list = bboxes[options.lpr.license_plate_name]
                    # 求相交同时置信度最高的车牌框
                    match_license_plate_roi = match_car_license_plate(bbox_info_idx['car_info']['roi'], license_plate_roi_list)

                    if len(match_license_plate_roi):
                        Latent_plate = match_license_plate_roi[0][0:4]

                        # 按照车牌宽高过滤车牌
                        if (Latent_plate[3] - Latent_plate[1] > options.lpr.plate_height[0]) and \
                            (Latent_plate[3] - Latent_plate[1] < options.lpr.plate_height[1]) and \
                            (Latent_plate[2] - Latent_plate[0] > options.lpr.plate_width[0]) and \
                            (Latent_plate[2] - Latent_plate[0] < options.lpr.plate_width[1]):

                            bbox_info_idx['plate_info']['roi'] = Latent_plate
                            
                            # lincense plate reader
                            # crop
                            x1 = min(max(0, int(bbox_info_idx['plate_info']['roi'][0] - options.lpr.lpr_ocr_width_expand_ratio * (bbox_info_idx['plate_info']['roi'][2] - bbox_info_idx['plate_info']['roi'][0]))), options.image_width)
                            x2 = min(max(0, int(bbox_info_idx['plate_info']['roi'][2] + options.lpr.lpr_ocr_width_expand_ratio * (bbox_info_idx['plate_info']['roi'][2] - bbox_info_idx['plate_info']['roi'][0]))), options.image_width)
                            y1 = min(max(0, int(bbox_info_idx['plate_info']['roi'][1])), options.image_height)
                            y2 = min(max(0, int(bbox_info_idx['plate_info']['roi'][3])), options.image_height)
                            # crop_img = img[bbox_info_idx['plate_info']['roi'][1]:bbox_info_idx['plate_info']['roi'][3], bbox_info_idx['plate_info']['roi'][0]:bbox_info_idx['plate_info']['roi'][2]]
                            crop_img = img[y1: y2, x1: x2]

                            # seg
                            seg_bbox, seg_info = self.lpr_seg.run(crop_img)

                            # ocr
                            # gray_img
                            if lpr.lpr_paddle_bool: 
                                plate_ocr, plate_scors_list = self.lpr.run(crop_img)
                            else:
                                gray_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                                plate_ocr, plate_scors_list = self.lpr.run(gray_crop_img)
                            

                            bbox_info_idx['plate_info']['num'] = plate_ocr
                            bbox_info_idx['plate_info']['score'] = np.array(plate_scors_list).mean()
                            bbox_info_idx['plate_info']['color'] = seg_info['color']

        return bbox_info_list

    def update_face_info( self, img, bboxes, bbox_info_list ):

        # 遍历单帧结果
        for idx in range(len(bbox_info_list)):
            bbox_info_idx = bbox_info_list[idx]

            if self.demo_type == "face":
                
                face_bbox_list = []
                for idx in range(len(options.face.sort_class_name)):
                    sort_class_name = options.face.sort_class_name[idx]
                    if sort_class_name in bboxes:
                        for idy in range(len(bboxes[sort_class_name])):
                            face_bbox_list.append([*bboxes[sort_class_name][idy], sort_class_name])

                # 求交集最大的框
                match_car_roi = match_bbox_iou(bbox_info_idx['face_info']['roi'], face_bbox_list)
                if len(match_car_roi):
                    bbox_info_idx['face_info']['roi'] = np.array(match_car_roi[0][0:4])
                    if face.landmark_bool:
                        bbox_info_idx['face_info']['landmark'] = np.array(match_car_roi[0][4:14])
                        bbox_info_idx['face_info']['landmark_degree'] = landmark2degree(copy.deepcopy(bbox_info_idx['face_info']['landmark']))
                    elif face.landmark_degree_bool:
                        bbox_info_idx['face_info']['landmark_degree'] = np.array(match_car_roi[0][5])
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
                        if options.lpr.sort_type == "car":
                            new_stable_loc = options.update_state_stable_loc_alpha * bbox_state_idy['state']['stable_loc'] +  (1 - options.update_state_stable_loc_alpha) * bbox_info_idx['car_info']['roi']
                    elif self.demo_type == "face":
                        bbox_state_idy['face_info']['roi'] = bbox_info_idx['face_info']['roi'] 
                        new_stable_loc = options.update_state_stable_loc_alpha * bbox_state_idy['state']['stable_loc'] +  (1 - options.update_state_stable_loc_alpha) * bbox_info_idx['face_info']['roi']

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
                        # 更新车牌识别有效帧数
                        bool_add_lpr = False
                        if bbox_info_idx['plate_info']['num'] != '' and \
                            bbox_state_idy['plate_info']['roi'][0] > options.ROI_Left_threshold and bbox_state_idy['plate_info']['roi'][2] < options.ROI_Right_threshold and \
                            bbox_state_idy['plate_info']['roi'][1] > options.ROI_Up_threshold and bbox_state_idy['plate_info']['roi'][3] < options.ROI_Down_threshold:
                            bool_add_lpr = True
                        
                        if bool_add_lpr:
                            bbox_state_idy['state']['obj_num'] += 1                   
                            bbox_state_idy['state']['obj_disappear_num'] = 0
                            bbox_state_idy['state']['lpr_num_list'].append(bbox_info_idx['plate_info']['num'])
                            bbox_state_idy['state']['lpr_score_list'].append(bbox_info_idx['plate_info']['score'])
                            bbox_state_idy['state']['lpr_color_list'].append(bbox_info_idx['plate_info']['color'])

                            if len( bbox_state_idy['state']['lpr_num_list'] ) > options.lpr_ocr_state_container_length:
                                bbox_state_idy['state']['lpr_num_list'].pop(0)

                            if len( bbox_state_idy['state']['lpr_score_list'] ) > options.lpr_ocr_state_container_length:
                                bbox_state_idy['state']['lpr_score_list'].pop(0)

                            if len( bbox_state_idy['state']['lpr_color_list'] ) > options.lpr_city_state_container_length:
                                bbox_state_idy['state']['lpr_color_list'].pop(0)
                        else:
                            bbox_state_idy['state']['obj_disappear_num'] += 1

                    elif self.demo_type == "face":
                        # 更新人脸识别有效帧数
                        bool_add_face = False
                        if bbox_state_idy['face_info']['roi'][0] > options.ROI_Left_threshold and bbox_state_idy['face_info']['roi'][2] < options.ROI_Right_threshold and \
                            bbox_state_idy['face_info']['roi'][1] > options.ROI_Up_threshold and bbox_state_idy['face_info']['roi'][3] < options.ROI_Down_threshold:
                            bool_add_face = True
                        
                        if bool_add_face:
                            bbox_state_idy['state']['obj_num'] += 1
                            bbox_state_idy['state']['obj_disappear_num'] = 0
                            bbox_state_idy['state']['face_landmark_degree_list'] = bbox_info_idx['face_info']['landmark_degree']
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
                    if options.lpr.sort_type == "car":
                        bbox_state_idy['state']['stable_loc'] = bbox_info_idx['car_info']['roi']

                    # 更新车牌识别有效帧数
                    bool_add_lpr = False
                    if bbox_info_idx['plate_info']['num'] != '' and \
                        bbox_state_idy['plate_info']['roi'][0] > options.ROI_Left_threshold and bbox_state_idy['plate_info']['roi'][2] < options.ROI_Right_threshold and \
                        bbox_state_idy['plate_info']['roi'][1] > options.ROI_Up_threshold and bbox_state_idy['plate_info']['roi'][3] < options.ROI_Down_threshold:
                        bool_add_lpr = True
                    
                    if bool_add_lpr:
                        bbox_state_idy['state']['obj_num'] += 1
                        bbox_state_idy['state']['obj_disappear_num'] = 0
                        bbox_state_idy['state']['lpr_num_list'].append(bbox_info_idx['plate_info']['num'])
                        bbox_state_idy['state']['lpr_score_list'].append(bbox_info_idx['plate_info']['score'])
                        bbox_state_idy['state']['lpr_color_list'].append(bbox_info_idx['plate_info']['color'])

                        if len( bbox_state_idy['state']['lpr_num_list'] ) > options.lpr_ocr_state_container_length:
                            bbox_state_idy['state']['lpr_num_list'].pop(0)
                        if len( bbox_state_idy['state']['lpr_color_list'] ) > options.lpr_city_state_container_length:
                            bbox_state_idy['state']['lpr_color_list'].pop(0)

                    else:
                        bbox_state_idy['state']['obj_disappear_num'] += 1

                elif self.demo_type == "face":
                    bbox_state_idy['face_info']['roi'] = bbox_info_idx['face_info']['roi']
                    bbox_state_idy['state']['stable_loc'] = bbox_info_idx['face_info']['roi']

                    # 更新人脸识别有效帧数
                    bool_add_face = False
                    if bbox_state_idy['face_info']['roi'][0] > options.ROI_Left_threshold and bbox_state_idy['face_info']['roi'][2] < options.ROI_Right_threshold and \
                        bbox_state_idy['face_info']['roi'][1] > options.ROI_Up_threshold and bbox_state_idy['face_info']['roi'][3] < options.ROI_Down_threshold:
                        bool_add_face = True
                    
                    if bool_add_face:
                        bbox_state_idy['state']['obj_num'] += 1
                        bbox_state_idy['state']['obj_disappear_num'] = 0
                        bbox_state_idy['state']['face_landmark_degree_list'] = bbox_info_idx['face_info']['landmark_degree']
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
            
            if bbox_state_idy['state']['obj_disappear_num'] == 0:
                
                if self.demo_type == "lpr":
                    loc_center_x = (bbox_state_idy['plate_info']['roi'][0] + bbox_state_idy['plate_info']['roi'][2]) / 2.0
                    loc_center_y = (bbox_state_idy['plate_info']['roi'][1] + bbox_state_idy['plate_info']['roi'][3]) / 2.0
                elif self.demo_type == "face":
                    loc_center_x = (bbox_state_idy['face_info']['roi'][0] + bbox_state_idy['face_info']['roi'][2]) / 2.0
                    loc_center_y = (bbox_state_idy['face_info']['roi'][1] + bbox_state_idy['face_info']['roi'][3]) / 2.0

                # 如果车辆向近处行驶, bbox_state_idy['state']['up_down_state_frame_num'] >= 3 条件用于避免刚进 ROI 或者车辆静止状态下的误判
                if bbox_state_idy['state']['up_down_state'] == 'Near' and bbox_state_idy['state']['up_down_state_frame_num'] >= 3:
                    if abs(loc_center_y - options.Down_threshold) < options.capture_plate_up_down_distance_boundary_threshold and \
                        bbox_state_idy['state']['obj_num'] > options.capture_plate_frame_threshold:
                        near_flage = True

                # 如果车辆向远处行驶，bbox_state_idy['state']['up_down_state_frame_num'] 条件用于避免刚进 ROI 或者车辆静止状态下的误判
                if bbox_state_idy['state']['up_down_state'] == 'Far' and bbox_state_idy['state']['up_down_state_frame_num'] >= 3:
                    if abs(loc_center_y - options.Up_threshold) < options.capture_plate_up_down_distance_boundary_threshold and \
                        bbox_state_idy['state']['obj_num'] > options.capture_plate_frame_threshold:
                        far_flage = True

                # 如果车辆向左边行驶
                if bbox_state_idy['state']['left_right_state'] == 'Left' and bbox_state_idy['state']['left_right_state_frame_num'] >= 3:
                    if (( loc_center_x - options.Left_threshold > 0 and \
                        loc_center_x - options.Left_threshold < options.capture_plate_left_right_distance_near_boundary_threshold ) or \
                        ( options.Left_threshold - loc_center_x > 0 and \
                        options.Left_threshold - loc_center_x < options.capture_plate_left_right_distance_far_boundary_threshold )) and \
                        bbox_state_idy['state']['obj_num'] > options.capture_plate_frame_threshold:
                        left_flage = True
                
                # 如果车辆向右边行驶
                if bbox_state_idy['state']['left_right_state'] == 'Right' and bbox_state_idy['state']['left_right_state_frame_num'] >= 3:
                    if (( loc_center_x - options.Right_threshold > 0 and \
                        loc_center_x - options.Right_threshold < options.capture_plate_left_right_distance_far_boundary_threshold ) or \
                        ( options.Right_threshold - loc_center_x > 0 and \
                        options.Right_threshold - loc_center_x < options.capture_plate_left_right_distance_near_boundary_threshold )) and \
                        bbox_state_idy['state']['obj_num'] > options.capture_plate_frame_threshold:
                        right_flage = True

                # 如果车辆在视野内，超过 50 帧
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
                        # capture_flage
                        lpr_num_np = np.array(bbox_state_idy['state']['lpr_num_list'])
                        lpr_score_np = np.array(bbox_state_idy['state']['lpr_score_list'])
                        lpr_color_np = np.array(bbox_state_idy['state']['lpr_color_list'])
                        lpr_color_np = lpr_color_np[lpr_color_np != "none"]

                        # 获得抓拍序列
                        if len(lpr_num_np[lpr_score_np > options.capture_lpr_score_threshold]):
                            capture_lpr_num, capture_lpr_num_frame = Counter(list(lpr_num_np[lpr_score_np > options.capture_lpr_score_threshold])).most_common(1)[0]
                            capture_from_container_list = self.find_capture_plate(bbox_state_idy['track_id'], capture_lpr_num)

                            if capture_lpr_num_frame >= options.capture_lpr_num_frame_threshold and \
                                len(capture_from_container_list) and \
                                capture_lpr_num not in self.params_dict['capture_res_container']:

                                capture_res_dict['track_id'] = capture_id_idx
                                capture_res_dict['capture']['flage'] = capture_dict_idy['capture']['flage']
                                capture_res_dict['capture']['img_bbox_info_list'] = capture_from_container_list
                                capture_res_dict['capture']['capture_bool'] = True

                                # num
                                capture_res_dict['plate_info']['num'] = capture_lpr_num
                                
                                # color
                                if lpr_color_np.shape[0]:
                                    capture_lpr_color, capture_lpr_color_frame = Counter(list(lpr_color_np)).most_common(1)[0]
                                    if capture_lpr_color_frame >= options.capture_lpr_color_frame_threshold:
                                        capture_res_dict['plate_info']['color'] = capture_lpr_color
                                    else:
                                        capture_res_dict['plate_info']['color'] = "none"
                                else:
                                    capture_res_dict['plate_info']['color'] = "none"                           
                                
                                self.params_dict['capture_res_container'][capture_res_dict['plate_info']['num']] = capture_res_dict

                                # 信息同步
                                capture_dict_idy['capture']['capture_bool'] = capture_res_dict['capture']['capture_bool']
                                capture_dict_idy['plate_info']['num'] = capture_res_dict['plate_info']['num']
                                capture_dict_idy['plate_info']['color'] = capture_res_dict['plate_info']['color']

                    elif self.demo_type == "face":
                        # 获得抓拍序列
                        capture_from_container_list = self.find_capture_face(bbox_state_idy['track_id'])

                        if len(capture_from_container_list) and \
                            capture_id_idx not in self.params_dict['capture_res_container']:

                            capture_res_dict['track_id'] = capture_id_idx
                            capture_res_dict['capture']['flage'] = capture_dict_idy['capture']['flage']
                            capture_res_dict['capture']['img_bbox_info_list'] = capture_from_container_list
                            capture_res_dict['capture']['capture_bool'] = True

                            self.params_dict['capture_res_container'][capture_res_dict['track_id']] = capture_res_dict

                            # 信息同步
                            capture_dict_idy['capture']['capture_bool'] = capture_res_dict['capture']['capture_bool']
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


    def find_capture_face(self, captute_id):
        
        capture_from_container_list = []

        for idy in range(len(self.params_dict['bbox_info_container'])):
            bbox_info_list = self.params_dict['bbox_info_container'][idy]['bbox_info']

            for idx in range(len(bbox_info_list)):
                bbox_info_idx = bbox_info_list[idx]

                # 容器中存在追踪对象
                if bbox_info_idx['track_id'] == captute_id and bbox_info_idx['face_info']['landmark_degree'] < options.capture_face_landmark_degree_threshold:
                    capture_res_info = copy.deepcopy(self.params_dict['bbox_info_container'][idy])
                    capture_res_info['bbox_info'] = [bbox_info_idx]
                    capture_from_container_list.append(capture_res_info)
        
        if len(capture_from_container_list) > 3:

            def sort_func(x):
                return x['bbox_info'][0]['face_info']['landmark_degree']

            capture_from_container_list = sorted(capture_from_container_list, key=sort_func)
            capture_from_container_list = capture_from_container_list[:3]
        
        return capture_from_container_list