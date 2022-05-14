from collections import Counter
import cv2
import numpy as np
import os
import sys 
import random
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo/Image')
# sys.path.insert(0, '/yuanhuan/code/demo/Image')
sys.path.insert(0, 'E:\\project\\demo\\Image')
from detection2d.ssd_rfb_crossdatatraining.test_tools import SSDDetector
from Basic.video_capture.plate_color.plate_color import update_plate_color
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


class VideoCaptureApi():
    """
    VideoCaptureApi
    """

    def __init__(self, model_prototxt, model_path, GPU):

        # init 
        self.model_prototxt = model_prototxt
        self.model_path = model_path
        self.GPU = GPU

        # option
        self.option_init()

        # param_init
        self.param_init()

        # model_init 
        self.model_init()
    
    
    def option_init(self):
        # detector

        # 2022-03-09-17
        # pytorch 
        # self.ssd_car_plate_prototxt = None
        # self.ssd_car_plate_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-03-09-17_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
        # caffe
        # self.ssd_car_plate_prototxt = "/mnt/huanyuan/model_final/image_model/ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-03-09-17/FPN_RFB_3class_3attri_noDilation_prior.prototxt"
        # self.ssd_car_plate_model_path = "/mnt/huanyuan/model_final/image_model/ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-03-09-17/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-03-09-17.caffemodel"
        
        # 2022-04-25-18
        # pytorch 
        # self.ssd_car_plate_prototxt = None
        # self.ssd_car_plate_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-04-25-18_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
        # caffe
        # self.ssd_car_plate_prototxt = "/mnt/huanyuan/model_final/image_model/ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-04-25-18/FPN_RFB_3class_3attri_noDilation_prior.prototxt"
        # self.ssd_car_plate_model_path = "/mnt/huanyuan/model_final/image_model/ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-04-25-18/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-04-25-18.caffemodel"
        # openvino
        # self.ssd_car_plate_prototxt = None
        # self.ssd_car_plate_model_path = "E:\\project\\model\\image\\ssd_rfb\\SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-04-25-18.xml"

        self.ssd_car_plate_prototxt = self.model_prototxt
        self.ssd_car_plate_model_path = self.model_path

        self.ssd_caffe_bool = False
        self.ssd_openvino_bool = True
        self.gpu_bool = self.GPU

        # 是否将 car\bus\truck 合并为一类输出
        self.merge_class_bool = None
        self.merge_class_name = 'car_bus_truck'
        self.car_attri_name_list = [ 'car', 'bus', 'truck' ]
        self.license_plate_name = 'license_plate'

        # 检测框下边界往下移动（检测框不准，车牌匹配不上）
        self.bbox_bottom_expand_bool = True
        self.bbox_bottom_expand = 50

        # sort
        self.max_age = 5 
        self.min_hits = 3 
        self.iou_threshold = 0.3

        # 状态容器长度
        self.bbox_state_container_length = 10       # 车辆框连续丢失上报，从容器中清除该车辆信息

        # 更新车辆行驶状态
        self.update_state_container_length = 10     # 车辆框坐标容器大小，用于判断车辆是否是处于静止状态
        self.update_state_num_threshold = 10        # 车辆行驶状态计数最大值，用于记录车辆处于同一行驶状态的帧数
        self.update_state_threshold = 1

        # 是否通过 roi 区域屏蔽部分检测结果
        # self.roi_bool = False
        self.roi_bool = True
        # 2M：16:9
        # args.roi_area = [0, 0, 1920, 1080]
        # 5M：16:9
        self.roi_area = [0, 462, 2592, 1920]

        # 抓拍线
        self.capture_line_ratio = [0.2, 0.8]


    def param_init(self):
        self.params_dict = {}

        # bbox_info_dict
        bbox_info_dict = {}
        bbox_info_dict['id'] = 0                                            # 追踪id
        bbox_info_dict['loc'] = []                                          # 车辆坐标
        bbox_info_dict['attri'] = 'None'                                    # 车辆属性：car, bus, truck
        bbox_info_dict['state'] = 'Stop'                                    # 车辆状态（上下行）
        bbox_info_dict['state_frame_num'] = 0                               # 车辆状态（上下行）帧数
        bbox_info_dict['frame_num'] = 0                                     # 车辆进入画面帧数
        bbox_info_dict['plate_loc'] = []                                    # 车牌坐标
        bbox_info_dict['plate_color'] = 'None'                              # 车牌颜色

        # bbox_state_dict
        bbox_state_dict = {}
        bbox_state_dict['id'] = 0                                           # 追踪id
        bbox_state_dict['loc'] = []                                         # 车辆坐标
        bbox_state_dict['loc_list'] = []                                    # 车辆坐标（多帧）
        bbox_state_dict['stable_loc'] = []                                  # 车辆坐标（稳定）
        bbox_state_dict['state'] = 'Stop'                                   # 车辆状态（上下行）
        bbox_state_dict['state_frame_num'] = 0                              # 车辆状态（上下行）帧数
        bbox_state_dict['start_frame'] = 0                                  # 车辆驶入画面的帧数
        bbox_state_dict['end_frame'] = 0                                    # 车辆驶出画面的帧数
        bbox_state_dict['frame_num'] = 0                                    # 车辆进入画面帧数
        bbox_state_dict['attri_list'] = []                                  # 车辆属性列表
        bbox_state_dict['plate_color_list'] = []                            # 车牌颜色列表
        bbox_state_dict['car_disappear_frame_num'] = 0                      # 车辆消失画面帧数
        bbox_state_dict['up_report_flage'] = False                          # 上行抓拍标志位
        bbox_state_dict['down_report_flage'] = False                        # 下行抓拍标志位

        self.params_dict['bbox_state_container'] = {}                       # 状态信息容器（key: 追踪id, value: bbox_state_dict）


    def model_init(self):
        # detector
        self.car_plate_detector = SSDDetector(prototxt=self.ssd_car_plate_prototxt, model_path=self.ssd_car_plate_model_path, ssd_caffe_bool=self.ssd_caffe_bool, ssd_openvino_bool=self.ssd_openvino_bool, merge_class_bool=self.merge_class_bool, gpu_bool = self.gpu_bool)

        # tracker
        self.mot_tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_threshold)
    

    def clear(self):
        # param_init
        self.param_init()


    def run(self, img, frame_idx):

        # info 
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]

        assert self.image_width == 2592
        assert self.image_height == 1920

        # detector
        bboxes = self.car_plate_detector.detect(img, with_score=True)

        # tracker 
        tracker_bboxes = self.update_tracker_bboxes(bboxes)

        # update bbox info
        bbox_info_list = self.update_bbox_info( img, bboxes, tracker_bboxes )

        # 更新状态容器，同时更新车辆行驶状态和帧率
        bbox_info_list = self.update_bbox_state_container( bbox_info_list, frame_idx )

        # captute
        capture_id_list = self.find_capture_id()
        capture_result = self.update_capture_state( capture_id_list, frame_idx )

        return bbox_info_list, capture_id_list, capture_result


    def update_tracker_bboxes(self, bboxes):
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
                    car_attri_name_idx = self.car_attri_name_list[idx]
                    if car_attri_name_idx in bboxes:
                        for idy in range(len(bboxes[car_attri_name_idx])):
                            bboxes[car_attri_name_idx][idy][3] = min(bboxes[car_attri_name_idx][idy][3] + self.bbox_bottom_expand, self.image_height)

            # tracker
            dets = np.empty((0, 5))
            for idx in range(len(self.car_attri_name_list)):
                car_attri_name_idx = self.car_attri_name_list[idx]
                if car_attri_name_idx in bboxes:
                    dets = np.concatenate((dets, np.array(bboxes[car_attri_name_idx])), axis=0)
                    
            tracker_bboxes = self.mot_tracker.update(dets)

        return tracker_bboxes


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

        bbox_info_list = []
        for idx in range(len(tracker_bboxes)):
            # init 
            # bbox_info_dict
            bbox_info_dict = {}
            bbox_info_dict['id'] = 0                                            # 追踪id
            bbox_info_dict['loc'] = []                                          # 车辆识别坐标
            bbox_info_dict['attri'] = 'None'                                    # 车辆属性：car, bus, truck
            bbox_info_dict['state'] = 'Stop'                                    # 车辆状态（上下行）
            bbox_info_dict['state_frame_num'] = 0                               # 车辆状态（上下行）帧数
            bbox_info_dict['frame_num'] = 0                                     # 车辆进入画面帧数
            bbox_info_dict['plate_loc'] = []                                    # 车牌坐标
            bbox_info_dict['plate_color'] = 'None'                              # 车牌颜色

            # car
            tracker_bbox = tracker_bboxes[idx]
            bbox_info_dict['id'] = tracker_bbox[-1]
            bbox_info_dict['loc'] = tracker_bbox[0:4]

            # 车辆属性更新
            if self.merge_class_bool:
                bbox_info_dict['attri'] = 'None'
            else:
                car_bbox_list = []
                for idx in range(len(self.car_attri_name_list)):
                    car_attri_name_idx = self.car_attri_name_list[idx]
                    if car_attri_name_idx in bboxes:
                        for idy in range(len(bboxes[car_attri_name_idx])):
                            car_bbox_list.append([*bboxes[car_attri_name_idx][idy], car_attri_name_idx])

                # 求交集最大的车辆框
                match_car_roi = self.match_bbox(bbox_info_dict['loc'], car_bbox_list)
                if len(match_car_roi):
                    bbox_info_dict['attri'] = match_car_roi[0][-1]

            # license plate
            if self.license_plate_name in bboxes:
                license_plate_roi_list = bboxes[self.license_plate_name]
                # 求交集最大的车牌框
                match_license_plate_roi = self.match_bbox(bbox_info_dict['loc'], license_plate_roi_list)

                if len(match_license_plate_roi):
                    bbox_info_dict['plate_loc'] = match_license_plate_roi[0][0:4]
                
                    # lincense plate reader
                    # crop
                    crop_img = img[bbox_info_dict['plate_loc'][1]:bbox_info_dict['plate_loc'][3], bbox_info_dict['plate_loc'][0]:bbox_info_dict['plate_loc'][2]]
                    bbox_info_dict['plate_color'] = update_plate_color(crop_img)

            bbox_info_list.append(bbox_info_dict)
        return bbox_info_list


    def update_bbox_state_container(self, bbox_info_list, frame_idx):

        # update
        pop_key_list = []
        for key, value in self.params_dict['bbox_state_container'].items():
            bbox_state_idy = value
            bbox_state_idy['car_disappear_frame_num'] += 1
            
            # pop
            if bbox_state_idy['car_disappear_frame_num'] >= self.bbox_state_container_length:
                pop_key_list.append(key)
        
        # pop
        for idx in range(len(pop_key_list)):
            self.params_dict['bbox_state_container'].pop(pop_key_list[idx])

        # 遍历单帧结果
        for idx in range(len(bbox_info_list)):
            bbox_info_idx = bbox_info_list[idx]

            is_new_state_bool = True

            # 遍历容器
            for key, value in self.params_dict['bbox_state_container'].items():
                bbox_state_idy = value

                # 容器中存在追踪对象
                if bbox_info_idx['id'] == bbox_state_idy['id']:

                    is_new_state_bool = False
                    bbox_state_idy['frame_num'] += 1
                    bbox_state_idy['loc'] = bbox_info_idx['loc']
                    bbox_state_idy['loc_list'].append(bbox_info_idx['loc'])
                    if len(bbox_state_idy['loc_list']) >= self.update_state_container_length: 
                        bbox_state_idy['loc_list'].pop(0)
                    bbox_state_idy['stable_loc'] = 0.9 * bbox_state_idy['stable_loc'] + 0.1 * bbox_info_idx['loc']
                    bbox_state_idy['car_disappear_frame_num'] = 0

                    # 更新车辆状态
                    car_center_y = ( np.array(bbox_state_idy['loc_list']).mean(0)[1] + np.array(bbox_state_idy['loc_list']).mean(0)[3] ) / 2
                    car_stable_center_y = ( bbox_state_idy['stable_loc'][1] + bbox_state_idy['stable_loc'][3] ) / 2
                    distance_y = car_stable_center_y - car_center_y

                    # 车辆状态判断
                    if distance_y > self.update_state_threshold:
                        bbox_state = 'Up'
                    elif distance_y < ( -1 * self.update_state_threshold ):
                        bbox_state = 'Down'
                    else:
                        bbox_state = "Stop"
                        
                    if bbox_state_idy['state'] != bbox_state:
                        if bbox_state_idy['state_frame_num'] > 0:
                            bbox_state_idy['state_frame_num'] -= 1
                        else:
                            bbox_state_idy['state'] = bbox_state
                            bbox_state_idy['state_frame_num'] = 0
                    else:
                        bbox_state_idy['state_frame_num'] = min( bbox_state_idy['state_frame_num'] + 1 , self.update_state_num_threshold)
                    
                    # 更新车牌颜色结果
                    if not bbox_info_idx['plate_loc'] == '':
                        if self.roi_bool:
                            # 进入 roi 区域，在记录车牌信息
                            # 原因：远处车牌太小，结果不可信
                            car_bottom_y = bbox_info_idx['loc'][3]
                            if car_bottom_y > self.roi_area[1]:
                                bbox_state_idy['plate_color_list'].append(bbox_info_idx['plate_color'])
                        else:
                            bbox_state_idy['plate_color_list'].append(bbox_info_idx['plate_color'])

                    bbox_info_idx['state'] = bbox_state_idy['state']
                    bbox_info_idx['state_frame_num'] = bbox_state_idy['state_frame_num']
                    bbox_info_idx['frame_num'] = bbox_state_idy['frame_num']
            
            if is_new_state_bool:

                # bbox_state_dict
                bbox_state_dict = {}
                bbox_state_dict['id'] = 0                                           # 追踪id
                bbox_state_dict['loc'] = []                                         # 车辆坐标
                bbox_state_dict['loc_list'] = []                                    # 车辆坐标（多帧）
                bbox_state_dict['stable_loc'] = []                                  # 车辆坐标（稳定）
                bbox_state_dict['state'] = 'Stop'                                   # 车辆状态（上下行）
                bbox_state_dict['state_frame_num'] = 0                              # 车辆状态（上下行）帧数
                bbox_state_dict['start_frame'] = 0                                  # 车辆驶入画面的帧数
                bbox_state_dict['end_frame'] = 0                                    # 车辆驶出画面的帧数
                bbox_state_dict['frame_num'] = 0                                    # 车辆进入画面帧数
                bbox_state_dict['attri_list'] = []                                  # 车辆属性列表
                bbox_state_dict['plate_color_list'] = []                            # 车牌颜色列表
                bbox_state_dict['car_disappear_frame_num'] = 0                      # 车辆消失画面帧数
                bbox_state_dict['up_report_flage'] = False                          # 上行抓拍标志位
                bbox_state_dict['down_report_flage'] = False                        # 下行抓拍标志位

                bbox_state_dict['id'] = bbox_info_idx['id']
                bbox_state_dict['loc'] = bbox_info_idx['loc']
                bbox_state_dict['loc_list'].append(bbox_info_idx['loc'])
                bbox_state_dict['stable_loc'] = bbox_info_idx['loc']
                bbox_state_dict['start_frame'] = frame_idx
                bbox_state_dict['attri_list'].append(bbox_info_idx['attri'])

                # 更新车牌颜色结果
                if not bbox_info_idx['plate_loc'] == '':
                    if self.roi_bool:
                        # 进入 roi 区域，在记录车牌信息
                        # 原因：远处车牌太小，结果不可信
                        car_bottom_y = bbox_info_idx['loc'][3]
                        if car_bottom_y > self.roi_area[1]:
                            bbox_state_dict['plate_color_list'].append(bbox_info_idx['plate_color'])
                    else:
                        bbox_state_dict['plate_color_list'].append(bbox_info_idx['plate_color'])

                self.params_dict['bbox_state_container'][bbox_state_dict['id']] = bbox_state_dict

        return bbox_info_list


    def find_capture_id(self):

        capture_id_list = []
        for key, value in self.params_dict['bbox_state_container'].items():
            bbox_state_idy = value
            car_bottom_y = bbox_state_idy['loc'][3]

            # 上下限阈值
            if self.roi_bool:
                Down_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_ratio[1]
                Up_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_ratio[0]
            else:
                Down_threshold = self.image_height * self.capture_line_ratio[1]
                Up_threshold = self.image_height * self.capture_line_ratio[0]

            if bbox_state_idy['state'] == 'Down' and bbox_state_idy['state_frame_num'] >= 3 and car_bottom_y > Down_threshold and not bbox_state_idy['down_report_flage']:
                capture_id_list.append((bbox_state_idy['id'], 'down_report_flage'))
            # elif bbox_state_idy['state'] == 'Up' and bbox_state_idy['state_frame_num'] >= 3 and car_bottom_y < Up_threshold and not bbox_state_idy['up_report_flage']:
            #     capture_id_list.append((bbox_state_idy['id'], 'up_report_flage'))
            else:
                pass
            
        return capture_id_list


    def update_capture_state(self, capture_id_list, frame_idx):
        
        capture_info_list = []
        for idx in range(len(capture_id_list)):
            capture_id_idx = capture_id_list[idx][0]
            capture_flage_idx = capture_id_list[idx][1]

            # init 
            capture_dict = {}
            capture_dict['id'] = capture_id_idx
            capture_dict['start_frame'] = 0
            capture_dict['end_frame'] = frame_idx
            capture_dict['attri'] = 'None'
            capture_dict['plate_color'] = 'None'

            for key, value in self.params_dict['bbox_state_container'].items():
                bbox_state_idy = value

                if bbox_state_idy['id'] == capture_id_idx:
                    
                    # capture_flage
                    if (not bbox_state_idy['down_report_flage'] and capture_flage_idx == 'down_report_flage'):
    
                        bbox_state_idy['down_report_flage'] = True

                        attri_np = np.array(bbox_state_idy['attri_list'])
                        plate_color_np = np.array(bbox_state_idy['plate_color_list'])

                        capture_attri, capture_attri_frame = Counter(list(attri_np)).most_common(1)[0]
                        capture_plate_color, capture_plate_color_frame = Counter(list(plate_color_np)).most_common(1)[0]

                        capture_dict['start_frame'] = bbox_state_idy['start_frame']
                        capture_dict['attri'] = capture_attri
                        capture_dict['plate_color'] = capture_plate_color
                        capture_info_list.append(capture_dict)

        return capture_info_list