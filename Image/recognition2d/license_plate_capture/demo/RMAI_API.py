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
from recognition2d.license_plate_recognition.infer.license_plate import license_palte_model_init_caffe, license_palte_crnn_recognition_caffe, license_palte_beamsearch_init, license_palte_crnn_recognition_beamsearch_caffe
from recognition2d.license_plate_capture.sort.mot_sort import Sort


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
        # detector
        self.ssd_car_plate_prototxt = None
        self.ssd_car_plate_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-03-09-17_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
        # self.ssd_car_plate_prototxt = "/mnt/huanyuan2/model/image_model/ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-03-09-17/FPN_RFB_3class_3attri_noDilation_prior.prototxt"
        # self.ssd_car_plate_model_path = "/mnt/huanyuan2/model/image_model/ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-03-09-17/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-03-09-17.caffemodel"
        self.ssd_caffe_bool = False

        # 是否将 car\bus\truck 合并为一类输出
        self.merge_class_bool = True
        self.merge_class_name = 'car_bus_truck'
        self.car_attri_name_list = [ 'car', 'bus', 'truck' ]
        self.license_plate_name = 'license_plate'

        # 检测框下边界往下移动（检测框不准，车牌匹配不上）
        self.bbox_bottom_expand_bool = False
        self.bbox_bottom_expand = 50

        # sort
        self.max_age = 5 
        self.min_hits = 3 
        self.iou_threshold = 0.3
        
        # lincense plate reader
        self.plate_recognition_prototxt = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china_softmax.prototxt"
        self.plate_recognition_model_path = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china.caffemodel"
        self.prefix_beam_search_bool = False

        # 缓存容器长度
        self.cache_container_length = 8

        # 缓存间隔
        self.cache_interval = 2

        # 更新车辆行驶状态
        self.update_state_interval = 5
        self.update_state_threshold = 10

        # 是否通过 roi 区域屏蔽部分检测结果
        # self.roi_bool = False
        self.roi_bool = True
        # 2M
        # args.roi_area = [0, 300, 1920, 1080]
        # 5M
        self.roi_area = [0, 600, 2592, 1920]

        # 抓拍线
        self.capture_line_ratio = [0.2, 0.8]
        self.capture_stop_frame_threshold = 200
        self.capture_plate_disappear_frame_threshold = 5
        self.capture_plate_ocr_score_threshold = 0.8
        self.capture_plate_ocr_frame_threshold = 4


    def param_init(self):
        self.params_dict = {}

        # bbox_info_dict
        bbox_info_dict = {}
        bbox_info_dict['id'] = 0                                            # 追踪id
        bbox_info_dict['loc'] = []                                          # 车辆坐标
        bbox_info_dict['attri'] = 'None'                                    # 车辆属性：car, bus, truck
        bbox_info_dict['state'] = ''                                        # 车辆状态（上下行）
        bbox_info_dict['frame_num'] = 0                                     # 车辆进入画面帧数
        bbox_info_dict['plate_loc'] = []                                    # 车牌坐标
        bbox_info_dict['plate_ocr'] = ''                                    # 车牌识别结果（单帧）
        bbox_info_dict['plate_ocr_score'] = 0.0                             # 车牌识别结果得分（单帧）
        
        self.params_dict['cache_container'] = []                            # 缓存容器（imag & bbox_info_dict）

        # bbox_state_dict
        bbox_state_dict = {}
        bbox_state_dict['id'] = 0                                           # 追踪id
        bbox_state_dict['loc'] = []                                         # 车辆坐标
        bbox_state_dict['init_center_y'] = 0.0                              # 车辆进入画面时的中心坐标
        bbox_state_dict['state'] = ''                                       # 车辆状态（上下行）
        bbox_state_dict['state_frame_num'] = ''                             # 车辆状态（上下行）帧数
        bbox_state_dict['frame_num'] = 0                                    # 车辆进入画面帧数
        bbox_state_dict['plate_ocr_list'] = []                              # 车牌识别结果（多帧）
        bbox_state_dict['plate_ocr_score_list'] = []                        # 车牌识别结果得分（多帧）
        bbox_state_dict['plate_disappear_frame_num'] = 0                    # 车辆消失画面帧数
        bbox_state_dict['capture_flage'] = False                            # 车辆抓拍标志位
        bbox_state_dict['outtime_flage'] = False                            # 车辆超时标志位
        
        self.params_dict['bbox_state_container'] = []                       # 状态信息容器（bbox_state_dict）


    def model_init(self):
        # detector
        self.car_plate_detector = SSDDetector(prototxt=self.ssd_car_plate_prototxt, model_path=self.ssd_car_plate_model_path, ssd_caffe_bool=self.ssd_caffe_bool, merge_class_bool=self.merge_class_bool)

        # tracker
        self.mot_tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_threshold)

        # lincense plate reader
        self.license_palte_reader = license_palte_model_init_caffe(self.plate_recognition_prototxt, self.plate_recognition_model_path)
        self.license_palte_beamsearch = license_palte_beamsearch_init()
    

    def clear(self):
        # param_init
        self.param_init()


    def run(self, img, frame_idx):

        # info 
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]

        # detector
        bboxes = self.car_plate_detector.detect(img, with_score=True)
        
        if self.merge_class_bool:
    
            # bboxes expand
            if self.bbox_bottom_expand_bool:
                if self.merge_class_name in bboxes:
                    for idx in range(len(bboxes[self.merge_class_name])):
                        bboxes[self.merge_class_name][idx][3] = min(bboxes[self.merge_class_name][idx][3] + self.bbox_bottom_expand, self.image_height)

            # tracker
            if self.merge_class_name in bboxes:
                dets = np.array(bboxes[self.merge_class_name])
            else:
                dets = np.empty((0, 5))
            tracker_bboxes = self.mot_tracker.update(dets)

        else:

            # bboxes expand
            if self.bbox_bottom_expand_bool:
                for idx in range(len(self.car_attri_name_list)):
                    car_name_idx = self.car_attri_name_list[idx]
                    if car_name_idx in bboxes:
                        for idy in range(len(bboxes[car_name_idx])):
                            bboxes[car_name_idx][idy][3] = min(bboxes[car_name_idx][idy][3] + self.bbox_bottom_expand, self.image_height)

            # tracker
            dets = np.empty((0, 5))
            for idx in range(len(self.car_attri_name_list)):
                car_name_idx = self.car_attri_name_list[idx]
                if car_name_idx in bboxes:
                    dets = np.concatenate((dets, np.array(bboxes[car_name_idx])), axis=0)
                    
            tracker_bboxes = self.mot_tracker.update(dets)

        # update bbox info
        bbox_info_list = self.update_bbox_info(img, bboxes, tracker_bboxes)

        # store
        # 跳帧存储原图和检测识别结果
        self.update_cache_container(img, frame_idx, bbox_info_list)

        # 更新状态容器，同时更新车辆行驶状态和帧率
        bbox_info_list = self.update_bbox_state_container(bbox_info_list)

        # captute
        ## capture_line
        if self.roi_bool:
            capture_line = [ self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * ratio for ratio in self.capture_line_ratio ]
        else:
            capture_line = [ self.image_height * ratio for ratio in self.capture_line_ratio ]

        capture_id_list = self.find_capture_id()
        capture_info = self.update_capture_state( capture_id_list )

        return bbox_info_list, capture_line, capture_id_list, capture_info


    def match_bbox(self, input_roi, match_roi_list):
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


    def update_bbox_info(self, img, bboxes, tracker_bboxes):
        # gray img for lincense plate reader
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bbox_info_list = []
        for idx in range(len(tracker_bboxes)):
            # init 
            # bbox_info_dict
            bbox_info_dict = {}
            bbox_info_dict['id'] = 0                                            # 追踪id
            bbox_info_dict['loc'] = []                                          # 车辆识别坐标
            bbox_info_dict['attri'] = 'None'                                    # 车辆属性：car, bus, truck
            bbox_info_dict['state'] = ''                                        # 车辆状态（上下行）
            bbox_info_dict['frame_num'] = 0                                     # 车辆进入画面帧数
            bbox_info_dict['plate_loc'] = []                                    # 车牌坐标
            bbox_info_dict['plate_ocr'] = ''                                    # 车牌识别结果（单帧）
            bbox_info_dict['plate_ocr_score'] = 0.0                             # 车牌识别结果得分（单帧）

            # car
            tracker_bbox = tracker_bboxes[idx]
            bbox_info_dict['id'] = tracker_bbox[-1]
            bbox_info_dict['loc'] = tracker_bbox[0:4]

            # 车辆属性更新
            if self.merge_class_bool:
                pass
            else:
                for idx in range(len(self.car_attri_name_list)):
                    car_name_idx = self.car_attri_name_list[idx]
                    if car_name_idx in bboxes:
                        car_roi_list = bboxes[car_name_idx]
                        # 求交集最大的车辆框
                        match_car_roi = self.match_bbox(bbox_info_dict['loc'], car_roi_list)
                        if len(match_car_roi):
                            bbox_info_dict['attri'] = car_name_idx
                        
            # license plate
            if self.license_plate_name in bboxes:
                license_plate_roi_list = bboxes[self.license_plate_name]
                # 求交集最大的车牌框
                match_license_plate_roi = self.match_bbox(bbox_info_dict['loc'], license_plate_roi_list)

                if len(match_license_plate_roi):
                    bbox_info_dict['plate_loc'] = match_license_plate_roi[0][0:4]

                    # lincense plate reader
                    # crop
                    crop_img = gray_img[bbox_info_dict['plate_loc'][1]:bbox_info_dict['plate_loc'][3], bbox_info_dict['plate_loc'][0]:bbox_info_dict['plate_loc'][2]]

                    if self.prefix_beam_search_bool:
                        # prefix beamsearch
                        _, plate_scors_list = license_palte_crnn_recognition_caffe(self.license_palte_reader, crop_img)
                        plate_ocr = license_palte_crnn_recognition_beamsearch_caffe(self.license_palte_reader, crop_img, self.license_palte_beamsearch[0], self.license_palte_beamsearch[1])
                    else:
                        # greedy
                        plate_ocr, plate_scors_list = license_palte_crnn_recognition_caffe(self.license_palte_reader, crop_img)
                    
                    bbox_info_dict['plate_ocr'] = plate_ocr
                    bbox_info_dict['plate_ocr_score'] = np.array(plate_scors_list).mean()
                
            bbox_info_list.append(bbox_info_dict)
        return bbox_info_list


    def update_cache_container(self, img, frame_idx, bbox_info_list):
        if frame_idx % self.cache_interval == 0:
            self.params_dict['cache_container'].append({'img': img, 'bbox_info': bbox_info_list})

        if len(self.params_dict['cache_container']) > self.cache_container_length:
            self.params_dict['cache_container'].pop(0)


    def update_bbox_state_container(self, bbox_info_list):
        
        # update
        for idy in range(len(self.params_dict['bbox_state_container'])):
            bbox_state_idy = self.params_dict['bbox_state_container'][idy]
            bbox_state_idy['plate_disappear_frame_num'] += 1
        
        is_new_bool = True
        # 遍历单帧结果
        for idx in range(len(bbox_info_list)):
            bbox_info_idx = bbox_info_list[idx]
            car_center_y = (bbox_info_idx['loc'][1] + bbox_info_idx['loc'][3]) / 2

            # 遍历容器
            for idy in range(len(self.params_dict['bbox_state_container'])):
                bbox_state_idy = self.params_dict['bbox_state_container'][idy]

                # 容器中存在追踪对象
                if bbox_info_idx['id'] == bbox_state_idy['id']:

                    is_new_bool = False
                    bbox_state_idy['frame_num'] += 1
                    bbox_state_idy['loc'] = bbox_info_idx['loc']

                    # 更新车辆状态
                    distance_y = bbox_state_idy['init_center_y'] - car_center_y
                    if distance_y > self.update_state_threshold:
                        bbox_state = 'Up'
                    elif distance_y < ( -1 * self.update_state_threshold ):
                        bbox_state = 'Down'
                    else:
                        bbox_state = "Stop"

                    if bbox_state_idy['state'] != bbox_state:
                        bbox_state_idy['state'] = bbox_state
                        bbox_state_idy['state_frame_num'] = 0
                    bbox_state_idy['state_frame_num'] += 1 
                    
                    # 更新车牌识别结果
                    if not bbox_info_idx['plate_ocr'] == '':
                        bbox_state_idy['plate_disappear_frame_num'] = 0

                        if self.roi_bool:
                            # 进入 roi 区域，在记录车牌信息
                            # 原因：远处车牌太小，结果不可信
                            car_bottom_y = bbox_info_idx['loc'][3]
                            if car_bottom_y > self.roi_area[1]:
                                bbox_state_idy['plate_ocr_list'].append(bbox_info_idx['plate_ocr'])
                                bbox_state_idy['plate_ocr_score_list'].append(bbox_info_idx['plate_ocr_score'])   
                        else:
                            bbox_state_idy['plate_ocr_list'].append(bbox_info_idx['plate_ocr'])
                            bbox_state_idy['plate_ocr_score_list'].append(bbox_info_idx['plate_ocr_score'])
                        

                    bbox_info_idx['state'] = bbox_state_idy['state']
                    bbox_info_idx['frame_num'] = bbox_state_idy['frame_num']
            
            if is_new_bool:

                # bbox_state_dict
                bbox_state_dict = {}
                bbox_state_dict['id'] = 0                                           # 追踪id
                bbox_state_dict['loc'] = []                                         # 车辆坐标
                bbox_state_dict['init_center_y'] = 0.0                              # 车辆进入画面时的中心坐标
                bbox_state_dict['state'] = ''                                       # 车辆状态（上下行）
                bbox_state_dict['state_frame_num'] = ''                             # 车辆状态（上下行）帧数
                bbox_state_dict['frame_num'] = 0                                    # 车辆进入画面帧数
                bbox_state_dict['plate_ocr_list'] = []                              # 车牌识别结果（多帧）
                bbox_state_dict['plate_ocr_score_list'] = []                        # 车牌识别结果得分（多帧）
                bbox_state_dict['plate_disappear_frame_num'] = 0                    # 车辆消失画面帧数
                bbox_state_dict['capture_flage'] = False                            # 车辆抓拍标志位
                bbox_state_dict['outtime_flage'] = False                            # 车辆超时标志位

                bbox_state_dict['id'] = bbox_info_idx['id']
                bbox_state_dict['loc'] = bbox_info_idx['loc']
                bbox_state_dict['init_center_y'] = car_center_y

                # 更新车牌识别结果
                if not bbox_info_idx['plate_ocr'] == '':
                    if self.roi_bool:
                        # 进入 roi 区域，在记录车牌信息
                        # 原因：远处车牌太小，结果不可信
                        car_bottom_y = bbox_info_idx['loc'][3]
                        if car_bottom_y > self.roi_area[1]:
                            bbox_state_dict['plate_ocr_list'].append(bbox_info_idx['plate_ocr'])
                            bbox_state_dict['plate_ocr_score_list'].append(bbox_info_idx['plate_ocr_score'])   
                    else:
                        bbox_state_dict['plate_ocr_list'].append(bbox_info_idx['plate_ocr'])
                        bbox_state_dict['plate_ocr_score_list'].append(bbox_info_idx['plate_ocr_score'])

                self.params_dict['bbox_state_container'].append(bbox_state_dict)

        return bbox_info_list


    def find_capture_id(self):

        capture_id_list = []
        for idy in range(len(self.params_dict['bbox_state_container'])):
            
            bbox_state_idy = self.params_dict['bbox_state_container'][idy]
            car_bottom_y = bbox_state_idy['loc'][3]

            # 上下限阈值
            if self.roi_bool:
                Down_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_ratio[1]
                Up_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_ratio[0]
            else:
                Down_threshold = self.image_height * self.capture_line_ratio[1]
                Up_threshold = self.image_height * self.capture_line_ratio[0]

            if bbox_state_idy['state'] == 'Down' and bbox_state_idy['state_frame_num'] >= 3 and car_bottom_y > Down_threshold and not bbox_state_idy['capture_flage']:
                capture_id_list.append((bbox_state_idy['id'], 'capture_flage'))
            elif bbox_state_idy['state'] == 'Up' and bbox_state_idy['state_frame_num'] >= 3 and car_bottom_y < Up_threshold and not bbox_state_idy['capture_flage']:
                capture_id_list.append((bbox_state_idy['id'], 'capture_flage'))
            elif bbox_state_idy['frame_num'] >= self.capture_stop_frame_threshold and not bbox_state_idy['outtime_flage']:
                capture_id_list.append((bbox_state_idy['id'], 'outtime_flage'))
            elif bbox_state_idy['plate_disappear_frame_num'] > self.capture_plate_disappear_frame_threshold and not bbox_state_idy['capture_flage']:
                capture_id_list.append((bbox_state_idy['id'], 'outtime_flage'))
            
        return capture_id_list


    def update_capture_state(self, capture_id_list):
        
        capture_info = []
        for idx in range(len(capture_id_list)):
            capture_id_idx = capture_id_list[idx][0]
            capture_flage_idx = capture_id_list[idx][1]

            # init 
            capture_dict = {}
            capture_dict['id'] = capture_id_idx
            capture_dict['plate_ocr'] = ''
            capture_dict['img_bbox_info'] = []

            for idy in range(len(self.params_dict['bbox_state_container'])):
                bbox_state_idy = self.params_dict['bbox_state_container'][idy]

                if bbox_state_idy['id'] == capture_id_idx:

                    # capture_flage
                    if (not bbox_state_idy['capture_flage'] and capture_flage_idx == 'capture_flage') or \
                        (not bbox_state_idy['outtime_flage'] and capture_flage_idx == 'outtime_flage'):

                        plate_ocr_np = np.array(bbox_state_idy['plate_ocr_list'])
                        plate_ocr_score_np = np.array(bbox_state_idy['plate_ocr_score_list'])

                        # 获得抓拍序列
                        if len(plate_ocr_np[plate_ocr_score_np > self.capture_plate_ocr_score_threshold]):
                            capture_license_palte, capture_license_palte_frame = Counter(list(plate_ocr_np[plate_ocr_score_np > self.capture_plate_ocr_score_threshold])).most_common(1)[0]
                            if capture_license_palte_frame >= self.capture_plate_ocr_frame_threshold:
                                capture_from_container_list = self.find_capture_plate(bbox_state_idy['id'], capture_license_palte)

                                # 抓到车牌，标志位置1
                                if len(capture_from_container_list) and capture_flage_idx == 'capture_flage':
                                    bbox_state_idy['capture_flage'] = True

                                # 无论是否抓到车牌，标志位置1
                                if len(capture_from_container_list) and capture_flage_idx == 'outtime_flage':
                                    bbox_state_idy['outtime_flage'] = True

                                capture_dict['plate_ocr'] = capture_license_palte
                                capture_dict['img_bbox_info'] = capture_from_container_list

                                capture_info.append(capture_dict)
        
        return capture_info


    def find_capture_plate(self, captute_id, capture_license_palte):
        
        capture_from_container_list = []

        for idy in range(len(self.params_dict['cache_container'])):
            bbox_info_list = self.params_dict['cache_container'][idy]['bbox_info']

            for idx in range(len(bbox_info_list)):
                bbox_info_idx = bbox_info_list[idx]

                # 容器中存在追踪对象
                if bbox_info_idx['id'] == captute_id and bbox_info_idx['plate_ocr'] == capture_license_palte:
                    capture_from_container_list.append(self.params_dict['cache_container'][idy])
        
        if len(capture_from_container_list) > 3:
            capture_from_container_list = random.sample(capture_from_container_list, 3)
        
        return capture_from_container_list
                        

                    

