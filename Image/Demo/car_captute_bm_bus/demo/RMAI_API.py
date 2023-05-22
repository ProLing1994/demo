from collections import Counter
import cv2
import numpy as np
import sys 
import random

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.detection2d.mmdetection.demo.detector.yolov6_detector import YOLOV6Detector
from Image.Demo.car_captute_bm_bus.sort.mot_sort import Sort


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

        self.image_width = 1920
        self.image_height = 1080

        # custom id 
        # self.custom_id = 1448
        self.custom_id = 3928

        # detector
        self.ssd_bool = False
        self.ssd_caffe_bool = False
        self.ssd_openvino_bool = False

        # # yolov6_c28_car_0214
        # self.yolov6_bool = True
        # self.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0214/yolov6_rm_c28.py"
        # self.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0214/epoch_135.pth"
        # self.yolov6_class_name = ['car']
        # self.yolov6_threshold_list = [0.3]

        # yolov6_c28_car_0320
        self.yolov6_bool = True
        # self.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0320/yolov6_rm_c28.py"
        # self.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0320/epoch_340.pth"
        self.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0320/yolov6_rm_c28_deploy.py"
        self.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0320/epoch_340_deploy.pth"
        self.yolov6_class_name = ['car']
        self.yolov6_threshold_list = [0.3]

        # sort
        self.max_age = 10
        self.min_hits = 3 
        self.iou_threshold = 0.3
        self.sort_class_name = 'car'

        # 状态容器长度
        self.bbox_state_container_length = 20       # 车辆框连续丢失上报，从容器中清除该车辆信息

        # 更新车辆行驶状态
        self.update_state_container_length = 1      # 车辆框坐标容器大小，用于判断车辆状态
        self.update_state_num_threshold = 5         # 车辆行驶状态计数最大值，用于记录车辆处于同一行驶状态的帧数
        self.update_state_threshold = 1 / 200
        self.update_state_stable_loc_alpha = float(0.6)   # 平滑车辆框参数

        # 抓拍区域
        if self.custom_id == 1448:
            self.capture_points = [(780, 190), (1070, 175), (1345, 190), (1730, 710), (330, 720)]
        elif self.custom_id == 3928:
            self.capture_points = [(780, 190), (1070, 175), (1345, 190), (1496, 404), (1730, 710), (330, 720), (613, 387)]

        self.alarm_left_threshold = [10, 200]
        self.alarm_right_threshold = [10, 200]
        self.alarm_top_threshold = [25, 200]
        self.alarm_lane_line_autio = 0.10

    def param_init(self):
        self.params_dict = {}

        # bbox_info_dict
        bbox_info_dict = {}
        bbox_info_dict['id'] = 0                                            # 追踪id
        bbox_info_dict['loc'] = []                                          # 车辆坐标
        bbox_info_dict['stable_loc'] = []                                   # 车辆坐标（稳定）
        bbox_info_dict['up_down_state'] = 'Stop'                            # 车辆状态（上下行）
        bbox_info_dict['up_down_state_frame_num'] = 0                       # 车辆状态（上下行）帧数
        bbox_info_dict['left_right_state'] = 'Stop'                         # 车辆状态（左右行）
        bbox_info_dict['left_right_state_frame_num'] = 0                    # 车辆状态（左右行）帧数
        bbox_info_dict['lane_line_info'] = []                               # 车道线状态
        bbox_info_dict['lane_line_state'] = '-1'                            # 车道线状态
        bbox_info_dict['lane_line_state_frame_num'] = 0                     # 车道线状态帧数
        bbox_info_dict['frame_num'] = 0                                     # 车辆进入画面帧数
        bbox_info_dict['up_down_speed'] = 0                                 # 车辆速度（上下行）
        bbox_info_dict['left_right_speed'] = 0                              # 车辆速度（左右行）
        bbox_info_dict['dist_capture_line_left'] = 0                        # 距离抓拍线位置
        bbox_info_dict['dist_capture_line_right'] = 0                       # 距离抓拍线位置
        bbox_info_dict['dist_capture_line_left_top'] = 0                    # 距离抓拍线位置
        bbox_info_dict['dist_capture_line_right_top'] = 0                   # 距离抓拍线位置

        bbox_info_dict['left_warning_state'] = 0                            # 抓拍标志位
        bbox_info_dict['left_alarm_flage'] = False                          # 抓拍标志位
        bbox_info_dict['left_in_alarm_flage'] = False                       # 抓拍标志位
        bbox_info_dict['left_out_alarm_flage'] = False                      # 抓拍标志位
        bbox_info_dict['right_warning_state'] = 0                           # 抓拍标志位
        bbox_info_dict['right_alarm_flage'] = False                         # 抓拍标志位
        bbox_info_dict['right_in_alarm_flage'] = False                      # 抓拍标志位
        bbox_info_dict['right_out_alarm_flage'] = False                     # 抓拍标志位
        bbox_info_dict['top_warning_state'] = 0                             # 抓拍标志位
        bbox_info_dict['top_alarm_flage'] = False                           # 抓拍标志位
        
        bbox_info_dict['warning_flage'] = False                             # 抓拍标志位
        bbox_info_dict['alarm_flage'] = False                               # 抓拍标志位
        bbox_info_dict['warning_cnt'] = 0                                   # 抓拍标志位
        bbox_info_dict['alarm_cnt'] = 0                                     # 抓拍标志位

        self.params_dict['cache_container'] = []                            # 缓存容器（imag & bbox_info_dict）

        # bbox_state_dict
        bbox_state_dict = {}
        bbox_state_dict['id'] = 0                                           # 追踪id
        bbox_state_dict['loc'] = []                                         # 车辆坐标
        bbox_state_dict['loc_list'] = []                                    # 车辆坐标（多帧）
        bbox_state_dict['stable_loc'] = []                                  # 车辆坐标（稳定）
        bbox_state_dict['up_down_state'] = 'Stop'                           # 车辆状态（上下行）
        bbox_state_dict['up_down_state_frame_num'] = 0                      # 车辆状态（上下行）帧数
        bbox_state_dict['left_right_state'] = 'Stop'                        # 车辆状态（左右行）
        bbox_state_dict['left_right_state_frame_num'] = 0                   # 车辆状态（左右行）帧数
        bbox_state_dict['lane_line_info'] = []                              # 车道线状态
        bbox_state_dict['lane_line_state'] = '-1'                           # 车道线状态
        bbox_state_dict['lane_line_state_frame_num'] = 0                    # 车道线状态帧数
        bbox_state_dict['frame_num'] = 0                                    # 车辆进入画面帧数
        bbox_state_dict['up_down_speed'] = 0                                # 车辆速度（上下行）
        bbox_state_dict['left_right_speed'] = 0                             # 车辆速度（左右行）
        bbox_state_dict['center_point_list'] = []                           # 车辆中心点轨迹（多帧）
        bbox_state_dict['car_disappear_frame_num'] = 0                      # 车辆消失画面帧数
        bbox_state_dict['dist_capture_line_left'] = 0                       # 距离抓拍线位置
        bbox_state_dict['dist_capture_line_right'] = 0                      # 距离抓拍线位置
        bbox_state_dict['dist_capture_line_left_top'] = 0                   # 距离抓拍线位置
        bbox_state_dict['dist_capture_line_right_top'] = 0                  # 距离抓拍线位置

        bbox_state_dict['left_warning_state'] = 0                           # 抓拍标志位
        bbox_state_dict['left_alarm_flage'] = False                         # 抓拍标志位
        bbox_state_dict['left_in_alarm_flage'] = False                      # 抓拍标志位
        bbox_state_dict['left_out_alarm_flage'] = False                     # 抓拍标志位
        bbox_state_dict['right_warning_state'] = 0                          # 抓拍标志位
        bbox_state_dict['right_alarm_flage'] = False                        # 抓拍标志位
        bbox_state_dict['right_in_alarm_flage'] = False                     # 抓拍标志位
        bbox_state_dict['right_out_alarm_flage'] = False                    # 抓拍标志位

        bbox_state_dict['top_warning_state'] = 0                            # 抓拍标志位
        bbox_state_dict['top_alarm_flage'] = False                          # 抓拍标志位
        bbox_state_dict['warning_flage'] = False                            # 抓拍标志位
        bbox_state_dict['alarm_flage'] = False                              # 抓拍标志位
        bbox_state_dict['warning_cnt'] = 0                                  # 抓拍标志位
        bbox_state_dict['alarm_cnt'] = 0                                    # 抓拍标志位

        self.params_dict['bbox_state_container'] = {}                       # 状态信息容器（key: 追踪id, value: bbox_state_dict）

    def model_init(self):
        # detector
        if self.yolov6_bool:
            self.detector = YOLOV6Detector(self.yolov6_config, self.yolov6_checkpoint, class_name=self.yolov6_class_name, threshold_list=self.yolov6_threshold_list, device='cuda:0')

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

        # 更新状态容器，同时更新车辆行驶状态和帧率
        bbox_info_list = self.update_bbox_state_container( bbox_info_list )
        bbox_state_map = self.params_dict['bbox_state_container']

        # captute
        capture_points =  self.capture_points

        self.update_capture()

        return tracker_bboxes, bbox_info_list, bbox_state_map, capture_points


    def update_tracker_bboxes(self, bboxes):
        
        # tracker
        if self.sort_class_name in bboxes:
            dets = np.array(bboxes[self.sort_class_name])
        else:
            dets = np.empty((0, 5))
                
        tracker_bboxes = self.mot_tracker.update(dets)

        return tracker_bboxes
    

    def update_bbox_info(self, img, bboxes, tracker_bboxes):

        bbox_info_list = []
        for idx in range(len(tracker_bboxes)):
            # init 
            # bbox_info_dict
            bbox_info_dict = {}
            bbox_info_dict['id'] = 0                                            # 追踪id
            bbox_info_dict['loc'] = []                                          # 车辆坐标
            bbox_info_dict['stable_loc'] = []                                   # 车辆坐标（稳定）
            bbox_info_dict['up_down_state'] = 'Stop'                            # 车辆状态（上下行）
            bbox_info_dict['up_down_state_frame_num'] = 0                       # 车辆状态（上下行）帧数
            bbox_info_dict['left_right_state'] = 'Stop'                         # 车辆状态（左右行）
            bbox_info_dict['left_right_state_frame_num'] = 0                    # 车辆状态（左右行）帧数
            bbox_info_dict['lane_line_state'] = '-1'                            # 车道线状态
            bbox_info_dict['lane_line_info'] = []                               # 车道线状态
            bbox_info_dict['lane_line_state_frame_num'] = 0                     # 车道线状态帧数
            bbox_info_dict['frame_num'] = 0                                     # 车辆进入画面帧数
            bbox_info_dict['up_down_speed'] = 0                                 # 车辆速度（上下行）
            bbox_info_dict['left_right_speed'] = 0                              # 车辆速度（左右行）
            bbox_info_dict['dist_capture_line_left'] = 0                        # 距离抓拍线位置
            bbox_info_dict['dist_capture_line_right'] = 0                       # 距离抓拍线位置
            bbox_info_dict['dist_capture_line_left_top'] = 0                    # 距离抓拍线位置
            bbox_info_dict['dist_capture_line_right_top'] = 0                   # 距离抓拍线位置
            bbox_info_dict['left_warning_state'] = 0                            # 抓拍标志位
            bbox_info_dict['left_alarm_flage'] = False                          # 抓拍标志位
            bbox_info_dict['left_in_alarm_flage'] = False                       # 抓拍标志位
            bbox_info_dict['left_out_alarm_flage'] = False                      # 抓拍标志位
            bbox_info_dict['right_warning_state'] = 0                           # 抓拍标志位
            bbox_info_dict['right_alarm_flage'] = False                         # 抓拍标志位
            bbox_info_dict['right_in_alarm_flage'] = False                      # 抓拍标志位
            bbox_info_dict['right_out_alarm_flage'] = False                     # 抓拍标志位
            bbox_info_dict['top_warning_state'] = 0                             # 抓拍标志位
            bbox_info_dict['top_alarm_flage'] = False                           # 抓拍标志位
            bbox_info_dict['warning_flage'] = False                             # 抓拍标志位
            bbox_info_dict['alarm_flage'] = False                               # 抓拍标志位
            bbox_info_dict['warning_cnt'] = 0                                   # 抓拍标志位
            bbox_info_dict['alarm_cnt'] = 0                                     # 抓拍标志位

            # car
            tracker_bbox = tracker_bboxes[idx]
            bbox_info_dict['id'] = tracker_bbox[-1]
            bbox_info_dict['loc'] = tracker_bbox[0:4]
    
            bbox_info_list.append(bbox_info_dict)

        return bbox_info_list


    def update_bbox_state_container(self, bbox_info_list):
        
        # update
        pop_key_list = []
        for key, bbox_state_idy in self.params_dict['bbox_state_container'].items():
          
            # pop
            if bbox_state_idy['car_disappear_frame_num'] > self.bbox_state_container_length:
                pop_key_list.append(key)
            
            bbox_state_idy['car_disappear_frame_num'] += 1
        
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
                if bbox_info_idx['id'] == bbox_state_idy['id']:

                    is_new_id_bool = False
                    bbox_state_idy['frame_num'] += 1
                    bbox_state_idy['loc'] = bbox_info_idx['loc']
                    bbox_state_idy['loc_list'].append(bbox_info_idx['loc'])
                    if len(bbox_state_idy['loc_list']) >= self.update_state_container_length: 
                        bbox_state_idy['loc_list'].pop(0)

                    # 更新车辆速度
                    new_stable_loc = self.update_state_stable_loc_alpha * bbox_state_idy['stable_loc'] +  (1 - self.update_state_stable_loc_alpha) * bbox_info_idx['loc']
                    old_stable_center_x = ( bbox_state_idy['stable_loc'][0] + bbox_state_idy['stable_loc'][2] ) / 2.0
                    new_stable_center_x = ( new_stable_loc[0] + new_stable_loc[2] ) / 2.0
                    old_stable_center_y = ( bbox_state_idy['stable_loc'][1] + bbox_state_idy['stable_loc'][3] ) / 2.0
                    new_stable_center_y = ( new_stable_loc[1] + new_stable_loc[3] ) / 2.0
                    bbox_state_idy['up_down_speed'] = (old_stable_center_y - new_stable_center_y) / float(bbox_state_idy['car_disappear_frame_num'])
                    bbox_state_idy['left_right_speed'] = (old_stable_center_x - new_stable_center_x) / float(bbox_state_idy['car_disappear_frame_num'])
                    bbox_state_idy['stable_loc'] = new_stable_loc

                    bbox_state_idy['car_disappear_frame_num'] = 0

                    bbox_w = bbox_state_idy['stable_loc'][2] - bbox_state_idy['stable_loc'][0]
                    bbox_h = bbox_state_idy['stable_loc'][3] - bbox_state_idy['stable_loc'][1]
                    # 车辆状态判断（上下行）
                    if bbox_state_idy['up_down_speed'] > self.update_state_threshold * bbox_h:
                        bbox_state = 'Far'
                    elif bbox_state_idy['up_down_speed'] < ( -1 * self.update_state_threshold * bbox_h ):
                        bbox_state = 'Near'
                    else:
                        bbox_state = "Stop"

                    if bbox_state_idy['up_down_state'] != bbox_state:
                        if bbox_state_idy['up_down_state_frame_num'] > 0:
                            bbox_state_idy['up_down_state_frame_num'] -= 1
                        else:
                            bbox_state_idy['up_down_state'] = bbox_state
                            bbox_state_idy['up_down_state_frame_num'] = 0
                    else:
                        bbox_state_idy['up_down_state_frame_num'] = min( bbox_state_idy['up_down_state_frame_num'] + 1 , self.update_state_num_threshold)
                    
                    # 车辆状态判断（左右行）
                    if bbox_state_idy['left_right_speed'] > self.update_state_threshold * bbox_w:
                        bbox_state = 'Left'
                    elif bbox_state_idy['left_right_speed'] < ( -1 * self.update_state_threshold * bbox_w):
                        bbox_state = 'Right'
                    else:
                        bbox_state = "Stop"

                    if bbox_state_idy['left_right_state'] != bbox_state:
                        if bbox_state_idy['left_right_state_frame_num'] > 0:
                            bbox_state_idy['left_right_state_frame_num'] -= 1
                        else:
                            bbox_state_idy['left_right_state'] = bbox_state
                            bbox_state_idy['left_right_state_frame_num'] = 0
                    else:
                        bbox_state_idy['left_right_state_frame_num'] = min( bbox_state_idy['left_right_state_frame_num'] + 1 , self.update_state_num_threshold)

                    # 抓拍线距离度量（左右抓拍线）
                    car_center_x = ( bbox_state_idy['stable_loc'][0] + bbox_state_idy['stable_loc'][2] ) / 2.0
                    car_center_y = ( bbox_state_idy['stable_loc'][1] + bbox_state_idy['stable_loc'][3] ) / 2.0
                    car_bottom_y = bbox_state_idy['stable_loc'][3]

                    # capture line left
                    if self.custom_id == 1448:
                        capture_point_left_top = self.capture_points[0]
                        capture_point_left_bottom = self.capture_points[4]
                    elif self.custom_id == 3928:
                        capture_point_left_top = self.capture_points[0]
                        capture_point_left_bottom = self.capture_points[5]
                    # y = kx + b
                    capture_line_left_k = float(capture_point_left_top[1] - capture_point_left_bottom[1]) / float(capture_point_left_top[0] - capture_point_left_bottom[0] + 1e-5);
                    capture_line_left_b = float(capture_point_left_top[1] - capture_point_left_top[0] * capture_line_left_k);
                    # dist
                    bbox_state_idy['dist_capture_line_left'] = (-1) * ((car_bottom_y - capture_line_left_b) / (capture_line_left_k + 1e-5) - car_center_x)

                    # capture line right
                    if self.custom_id == 1448:
                        capture_point_right_top = self.capture_points[2]
                        capture_point_right_bottom = self.capture_points[3]
                    elif self.custom_id == 3928:
                        capture_point_right_top = self.capture_points[2]
                        capture_point_right_bottom = self.capture_points[4]
                    # y = kx + b
                    capture_line_right_k = float(capture_point_right_top[1] - capture_point_right_bottom[1]) / float(capture_point_right_top[0] - capture_point_right_bottom[0] + 1e-5);
                    capture_line_right_b = float(capture_point_right_top[1] - capture_point_right_top[0] * capture_line_right_k);
                    # dist
                    bbox_state_idy['dist_capture_line_right'] = (-1) * ((car_bottom_y - capture_line_right_b) / (capture_line_right_k + 1e-5) - car_center_x)

                    # 抓拍线距离度量（上下抓拍线）
                    # capture line left top
                    capture_point_left_top = self.capture_points[0]
                    capture_point_middle_top = self.capture_points[1]
                    # y = kx + b
                    capture_line_left_top_k = float(capture_point_middle_top[1] - capture_point_left_top[1]) / float(capture_point_middle_top[0] - capture_point_left_top[0] + 1e-5);
                    capture_line_left_top_b = float(capture_point_left_top[1] - capture_point_left_top[0] * capture_line_left_top_k);
                    # dist
                    bbox_state_idy['dist_capture_line_left_top'] = (-1) * ( car_center_x * (capture_line_left_top_k + 1e-5) + capture_line_left_top_b - car_bottom_y )

                    # capture line right top
                    capture_point_middle_top = self.capture_points[1]
                    capture_point_right_top = self.capture_points[2]
                    # y = kx + b
                    capture_line_right_top_k = float(capture_point_right_top[1] - capture_point_middle_top[1]) / float(capture_point_right_top[0] - capture_point_middle_top[0] + 1e-5);
                    capture_line_right_top_b = float(capture_point_right_top[1] - capture_point_right_top[0] * capture_line_right_top_k);
                    # dist
                    bbox_state_idy['dist_capture_line_right_top'] = (-1) * ( car_center_x * (capture_line_right_top_k + 1e-5) + capture_line_right_top_b - car_bottom_y )

                    # 车道线度量
                    # 记录车道线
                    # [[point_1, point_2, point_k, point_b, point_intersect], ]
                    bbox_state_idy['lane_line_info'] = []
                    bbox_state_idy['lane_line_info'] = []
                    if self.custom_id == 1448:
                        bbox_state_idy['lane_line_info'].append([self.capture_points[0], self.capture_points[2], 0, 0, [0, 0]])
                        bbox_state_idy['lane_line_info'].append([self.capture_points[4], self.capture_points[3], 0, 0, [0, 0]])
                    elif self.custom_id == 3928:
                        bbox_state_idy['lane_line_info'].append([self.capture_points[0], self.capture_points[2], 0, 0, [0, 0]])
                        bbox_state_idy['lane_line_info'].append([self.capture_points[6], self.capture_points[3], 0, 0, [0, 0]])
                        bbox_state_idy['lane_line_info'].append([self.capture_points[5], self.capture_points[4], 0, 0, [0, 0]])

                    # 更新车道线截距斜率
                    for lane_line_idx in range(len(bbox_state_idy['lane_line_info'])):
                        point_1 = bbox_state_idy['lane_line_info'][lane_line_idx][0]
                        point_2 = bbox_state_idy['lane_line_info'][lane_line_idx][1]

                        point_k = float(point_1[1] - point_2[1]) / float(point_1[0] - point_2[0] + 1e-5);
                        point_b = float(point_1[1] - point_1[0] * point_k);

                        bbox_state_idy['lane_line_info'][lane_line_idx][2] = point_k
                        bbox_state_idy['lane_line_info'][lane_line_idx][3] = point_b
                    
                    if ( bbox_state_idy['dist_capture_line_right'] < 0 and bbox_state_idy['dist_capture_line_left'] > 0 ):

                        # 更新车道线交点坐标
                        ## 获得平行线(k_intersect, b_intersect)
                        if abs(bbox_state_idy['dist_capture_line_right']) < abs(bbox_state_idy['dist_capture_line_left']):
                            k_intersect = capture_line_right_k
                            b_intersect = car_bottom_y - (capture_line_right_k * car_center_x)
                        else:
                            k_intersect = capture_line_left_k
                            b_intersect = car_bottom_y - (capture_line_left_k * car_center_x)

                        ## 获得交点坐标
                        for lane_line_idx in range(len(bbox_state_idy['lane_line_info'])):
                            point_k = bbox_state_idy['lane_line_info'][lane_line_idx][2]
                            point_b = bbox_state_idy['lane_line_info'][lane_line_idx][3]

                            point_intersect_x = (point_b - b_intersect)/(k_intersect - point_k)
                            point_intersect_y = k_intersect * point_intersect_x + b_intersect

                            bbox_state_idy['lane_line_info'][lane_line_idx][4] = [int(point_intersect_x + 0.5), int(point_intersect_y + 0.5)]
                        
                        ## 获得车道线
                        bbox_lane_line = -1
                        for lane_line_idx in range(len(bbox_state_idy['lane_line_info']) - 1):
                            lane_line_num = len(bbox_state_idy['lane_line_info']) - 1 - lane_line_idx
                            point_intersect_top = bbox_state_idy['lane_line_info'][lane_line_idx][4]
                            point_intersect_bottom = bbox_state_idy['lane_line_info'][lane_line_idx + 1][4]

                            if (car_bottom_y > point_intersect_top[1] and car_bottom_y <= point_intersect_bottom[1]):
                                # # 如果 （车辆底边位置 y 到 车道线上沿 长度）/ 车道线长度 > 0.95，则认为是下一个车道
                                # # 如果 （车辆底边位置 y 到 车道线上沿 长度）/ 车道线长度 <= 0.95，则认为是当前车道
                                # if ((car_bottom_y - point_intersect_top[1]) / (point_intersect_bottom[1] - point_intersect_top[1])) > (1 - self.alarm_lane_line_autio):
                                #     bbox_lane_line = max(1, lane_line_num - 1)
                                # else:
                                #     bbox_lane_line = lane_line_num

                                # 如果 （车辆底边位置 y 到 车道线上沿 长度）/ 车道线长度 > 0.1，则认为是当前车道
                                # 如果 （车辆底边位置 y 到 车道线上沿 长度）/ 车道线长度 <= 0.1，则认为是上一个车道
                                if ((car_bottom_y - point_intersect_top[1]) / (point_intersect_bottom[1] - point_intersect_top[1])) > self.alarm_lane_line_autio:
                                    bbox_lane_line = lane_line_num
                                else:
                                    bbox_lane_line = min((len(bbox_state_idy['lane_line_info']) - 1), lane_line_num + 1)

                            # 边缘车道特殊处理，防止出界
                            elif lane_line_num == (len(bbox_state_idy['lane_line_info']) - 1) and car_bottom_y <= point_intersect_top[1]:
                                if (-1 * (car_bottom_y - point_intersect_top[1])) / (point_intersect_bottom[1] - point_intersect_top[1]) < self.alarm_lane_line_autio:
                                    bbox_lane_line = lane_line_num
                            
                            # 边缘车道特殊处理，防止出界
                            elif lane_line_num == 1 and car_bottom_y > point_intersect_bottom[1]:
                                bbox_lane_line = lane_line_num

                        # 更新车道状态
                        if bbox_state_idy['lane_line_state'] != bbox_lane_line:
                            if bbox_state_idy['lane_line_state_frame_num'] > 0:
                                bbox_state_idy['lane_line_state_frame_num'] -= 1
                            else:
                                bbox_state_idy['lane_line_state'] = bbox_lane_line
                                bbox_state_idy['lane_line_state_frame_num'] = 0
                        else:
                            bbox_state_idy['lane_line_state_frame_num'] = min( bbox_state_idy['lane_line_state_frame_num'] + 1 , self.update_state_num_threshold)

                    bbox_info_idx['up_down_state'] = bbox_state_idy['up_down_state']
                    bbox_info_idx['up_down_state_frame_num'] = bbox_state_idy['up_down_state_frame_num']
                    bbox_info_idx['left_right_state'] = bbox_state_idy['left_right_state']
                    bbox_info_idx['left_right_state_frame_num'] = bbox_state_idy['left_right_state_frame_num']
                    bbox_info_idx['lane_line_info'] = bbox_state_idy['lane_line_info']
                    bbox_info_idx['lane_line_state'] = bbox_state_idy['lane_line_state']
                    bbox_info_idx['lane_line_state_frame_num'] = bbox_state_idy['lane_line_state_frame_num']
                    bbox_info_idx['stable_loc'] = bbox_state_idy['stable_loc']
                    bbox_info_idx['frame_num'] = bbox_state_idy['frame_num']
                    bbox_info_idx['up_down_speed'] = bbox_state_idy['up_down_speed']
                    bbox_info_idx['left_right_speed'] = bbox_state_idy['left_right_speed']
                    bbox_info_idx['dist_capture_line_left'] = bbox_state_idy['dist_capture_line_left']
                    bbox_info_idx['dist_capture_line_right'] = bbox_state_idy['dist_capture_line_right']
                    bbox_info_idx['dist_capture_line_left_top'] = bbox_state_idy['dist_capture_line_left_top']
                    bbox_info_idx['dist_capture_line_right_top'] = bbox_state_idy['dist_capture_line_right_top']
                    bbox_info_idx['left_warning_state'] = bbox_state_idy['left_warning_state']
                    bbox_info_idx['left_alarm_flage'] = bbox_state_idy['left_alarm_flage']
                    bbox_info_idx['left_in_alarm_flage'] = bbox_state_idy['left_in_alarm_flage']
                    bbox_info_idx['left_out_alarm_flage'] = bbox_state_idy['left_out_alarm_flage']
                    bbox_info_idx['right_warning_state'] = bbox_state_idy['right_warning_state']
                    bbox_info_idx['right_alarm_flage'] = bbox_state_idy['right_alarm_flage']
                    bbox_info_idx['right_in_alarm_flage'] = bbox_state_idy['right_in_alarm_flage']
                    bbox_info_idx['right_out_alarm_flage'] = bbox_state_idy['right_out_alarm_flage']
                    bbox_info_idx['top_warning_state'] = bbox_state_idy['top_warning_state']
                    bbox_info_idx['top_alarm_flage'] = bbox_state_idy['top_alarm_flage']
                    bbox_info_idx['warning_flage'] = bbox_state_idy['warning_flage']
                    bbox_info_idx['alarm_flage'] = bbox_state_idy['alarm_flage']
                    bbox_info_idx['warning_cnt'] = bbox_state_idy['warning_cnt']
                    bbox_info_idx['alarm_cnt'] = bbox_state_idy['alarm_cnt']

            if is_new_id_bool:

                # bbox_state_dict
                bbox_state_dict = {}
                bbox_state_dict['id'] = 0                                           # 追踪id
                bbox_state_dict['loc'] = []                                         # 车辆坐标
                bbox_state_dict['loc_list'] = []                                    # 车辆坐标（多帧）
                bbox_state_dict['stable_loc'] = []                                  # 车辆坐标（稳定）
                bbox_state_dict['up_down_state'] = 'Stop'                           # 车辆状态（上下行）
                bbox_state_dict['up_down_state_frame_num'] = 0                      # 车辆状态（上下行）帧数
                bbox_state_dict['left_right_state'] = 'Stop'                        # 车辆状态（左右行）
                bbox_state_dict['left_right_state_frame_num'] = 0                   # 车辆状态（左右行）帧数
                bbox_state_dict['lane_line_state'] = '-1'                           # 车道线状态
                bbox_state_dict['lane_line_info'] = []                              # 车道线状态
                bbox_state_dict['lane_line_state_frame_num'] = 0                    # 车道线状态帧数
                bbox_state_dict['frame_num'] = 0                                    # 车辆进入画面帧数
                bbox_state_dict['up_down_speed'] = 0                                # 车辆速度（上下行）
                bbox_state_dict['left_right_speed'] = 0                             # 车辆速度（左右行）
                bbox_state_dict['center_point_list'] = []                           # 车辆中心点轨迹（多帧）
                bbox_state_dict['car_disappear_frame_num'] = 0                      # 车辆消失画面帧数
                bbox_state_dict['dist_capture_line_left'] = 0                       # 距离抓拍线位置
                bbox_state_dict['dist_capture_line_right'] = 0                      # 距离抓拍线位置
                bbox_state_dict['dist_capture_line_left_top'] = 0                   # 距离抓拍线位置
                bbox_state_dict['dist_capture_line_right_top'] = 0                  # 距离抓拍线位置
                bbox_state_dict['left_warning_state'] = 0                           # 抓拍标志位
                bbox_state_dict['left_alarm_flage'] = False                         # 抓拍标志位
                bbox_state_dict['left_in_alarm_flage'] = False                      # 抓拍标志位
                bbox_state_dict['left_out_alarm_flage'] = False                     # 抓拍标志位
                bbox_state_dict['right_warning_state'] = 0                          # 抓拍标志位
                bbox_state_dict['right_alarm_flage'] = False                        # 抓拍标志位
                bbox_state_dict['right_in_alarm_flage'] = False                     # 抓拍标志位
                bbox_state_dict['right_out_alarm_flage'] = False                    # 抓拍标志位
                bbox_state_dict['top_warning_state'] = 0                            # 抓拍标志位
                bbox_state_dict['top_alarm_flage'] = False                          # 抓拍标志位
                bbox_state_dict['warning_flage'] = False                            # 抓拍标志位
                bbox_state_dict['alarm_flage'] = False                              # 抓拍标志位
                bbox_state_dict['warning_cnt'] = 0                                  # 抓拍标志位
                bbox_state_dict['alarm_cnt'] = 0                                    # 抓拍标志位

                bbox_state_dict['id'] = bbox_info_idx['id']
                bbox_state_dict['loc'] = bbox_info_idx['loc']
                bbox_state_dict['loc_list'].append(bbox_info_idx['loc'])
                bbox_state_dict['stable_loc'] = bbox_info_idx['loc']

                bbox_info_idx['stable_loc'] = bbox_info_idx['loc']

                self.params_dict['bbox_state_container'][bbox_state_dict['id']] = bbox_state_dict

        # update
        # center_point_list
        pop_key_list = []
        for key, bbox_state_idy in self.params_dict['bbox_state_container'].items():

            bbox_state_idy['center_point_list'].append( ( (bbox_state_idy['loc'][0] + bbox_state_idy['loc'][2]) / 2 , (bbox_state_idy['loc'][1] + bbox_state_idy['loc'][3]) / 2 ) )
            
            if len( bbox_state_idy['center_point_list'] ) > self.bbox_state_container_length:
                bbox_state_idy['center_point_list'].pop(0)

        return bbox_info_list
    

    def update_capture(self):

        # 报警逻辑
        # 1、车辆从左侧驶入 -> 右侧驶出
        # 2、车辆从右侧驶入 -> 左侧驶出
        # 3、车辆从上方驶入 -> 左侧驶出
        # 4、车辆从上方驶入 -> 右侧驶出
        # 5、车辆从左侧驶入 -> 上方驶出
        # 6、车辆从右侧驶入 -> 上方驶出

        for _, bbox_state_idy in self.params_dict['bbox_state_container'].items():

            car_left_x = bbox_state_idy['loc'][0]
            car_right_x = bbox_state_idy['loc'][2]
            car_bottom_y = bbox_state_idy['loc'][3]

            if self.custom_id == 1448:
                capture_point_left_top = self.capture_points[0]
                capture_point_left_bottom = self.capture_points[4]
                capture_point_right_top = self.capture_points[2]
                capture_point_right_bottom = self.capture_points[3]
                capture_point_middle_top = self.capture_points[1]
            elif self.custom_id == 3928:
                capture_point_left_top = self.capture_points[0]
                capture_point_left_bottom = self.capture_points[5]
                capture_point_right_top = self.capture_points[2]
                capture_point_right_bottom = self.capture_points[4]
                capture_point_middle_top = self.capture_points[1]

            # 如果车辆向右边行驶/向左边行驶
            # left_warning/left_alarm
            if car_bottom_y > capture_point_left_top[1]:
                if bbox_state_idy['dist_capture_line_left'] > ( -1 * self.alarm_left_threshold[1] ) and \
                    bbox_state_idy['dist_capture_line_left'] < ( -1 * self.alarm_left_threshold[0] ) : 

                    if bbox_state_idy['left_warning_state'] == 2:
                        bbox_state_idy['left_alarm_flage'] = True 
                        bbox_state_idy['left_out_alarm_flage'] = True 
                    else:
                        bbox_state_idy['left_warning_state'] = 1

                if bbox_state_idy['dist_capture_line_left'] > self.alarm_left_threshold[0] and \
                    bbox_state_idy['dist_capture_line_left'] < self.alarm_left_threshold[1]:

                    if bbox_state_idy['left_warning_state'] == 1:
                        bbox_state_idy['left_alarm_flage'] = True 
                        bbox_state_idy['left_in_alarm_flage'] = True
                    else:
                        bbox_state_idy['left_warning_state'] = 2

            # right_warning/right_alarm
            if car_bottom_y > capture_point_right_top[1]:
                if bbox_state_idy['dist_capture_line_right'] > ( -1 * self.alarm_right_threshold[1]) and \
                    bbox_state_idy['dist_capture_line_right'] < ( -1 * self.alarm_right_threshold[0]): 

                    if bbox_state_idy['right_warning_state'] == 2:
                        bbox_state_idy['right_alarm_flage'] = True 
                        bbox_state_idy['right_in_alarm_flage'] = True 
                    else:
                        bbox_state_idy['right_warning_state'] = 1

                if bbox_state_idy['dist_capture_line_right'] > self.alarm_right_threshold[0] and \
                    bbox_state_idy['dist_capture_line_right'] < self.alarm_right_threshold[1]:

                    if bbox_state_idy['right_warning_state'] == 1:
                        bbox_state_idy['right_alarm_flage'] = True 
                        bbox_state_idy['right_out_alarm_flage'] = True
                    else:
                        bbox_state_idy['right_warning_state'] = 2

            # 如果车辆向远处行驶
            # left_top_warning / left_top_alarm
            if (car_right_x > capture_point_left_top[0] and car_left_x <= capture_point_middle_top[0]):
                if bbox_state_idy['dist_capture_line_left_top'] > ( -1 * self.alarm_top_threshold[1]) and \
                    bbox_state_idy['dist_capture_line_left_top'] < ( -1 * self.alarm_top_threshold[0]) : 

                    if bbox_state_idy['top_warning_state'] == 2:
                        bbox_state_idy['top_alarm_flage'] = True 
                    else:
                        bbox_state_idy['top_warning_state'] = 1
            
                if bbox_state_idy['dist_capture_line_left_top'] > self.alarm_top_threshold[0] and \
                    bbox_state_idy['dist_capture_line_left_top'] < self.alarm_top_threshold[1] :

                    if bbox_state_idy['top_warning_state'] == 1: 
                        bbox_state_idy['top_alarm_flage'] = True 
                    else:
                        bbox_state_idy['top_warning_state'] = 2

            # right_top_warning / right_top_alarm
            if (car_right_x > capture_point_middle_top[0] and car_left_x < capture_point_right_top[0]):
                if bbox_state_idy['dist_capture_line_right_top'] > ( -1 * self.alarm_top_threshold[1]) and \
                    bbox_state_idy['dist_capture_line_right_top'] < ( -1 * self.alarm_top_threshold[0]) : 

                    if bbox_state_idy['top_warning_state'] == 2:
                        bbox_state_idy['top_alarm_flage'] = True 
                    else:
                        bbox_state_idy['top_warning_state'] = 1
            
                if bbox_state_idy['dist_capture_line_right_top'] > self.alarm_top_threshold[0] and \
                    bbox_state_idy['dist_capture_line_right_top'] < self.alarm_top_threshold[1] :

                    if bbox_state_idy['top_warning_state'] == 1: 
                        bbox_state_idy['top_alarm_flage'] = True 
                    else:
                        bbox_state_idy['top_warning_state'] = 2
                        
            # warning_flage
            if bbox_state_idy['left_alarm_flage'] or bbox_state_idy['right_alarm_flage'] or bbox_state_idy['top_alarm_flage']:

                bbox_state_idy['warning_cnt'] += 1
                bbox_state_idy['warning_flage'] = True
            
            # alarm_flage
            if bbox_state_idy['left_alarm_flage'] + bbox_state_idy['right_alarm_flage'] + bbox_state_idy['top_alarm_flage'] >= 2 or \
                bbox_state_idy['left_out_alarm_flage'] or bbox_state_idy['right_out_alarm_flage']:

                bbox_state_idy['alarm_cnt'] += 1
                bbox_state_idy['alarm_flage'] = True
        