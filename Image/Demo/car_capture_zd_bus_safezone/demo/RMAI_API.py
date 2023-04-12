from collections import Counter
import cv2
import numpy as np
import sys 
import random

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.detection2d.mmdetection.demo.detector.yolov6_detector import YOLOV6Detector
from Image.Demo.car_capture_zd_bus_safezone.sort.mot_sort import Sort


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
        self.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0320/yolov6_rm_c28.py"
        self.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0320/epoch_340.pth"
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

        # # 抓拍区域
        # self.capture_points = [(244, 973), (691, 687), (233, 842), (623, 605), (232, 695), (557, 529), 
        #                        (243, 633), (529, 494), (256, 561), (510, 449), (261, 506), (484, 411), 
        #                        (265, 467), (467, 383), (266, 423), (456, 358), (272, 389), (439, 329)]
        # 抓拍区域
        self.capture_points = [(156, 994), (567, 724), (148, 827), (515, 616), (144, 723), (491, 554), 
                               (162, 638), (471, 504), (181, 552), (460, 433), (199, 513), (438, 408), 
                               (219, 468), (428, 382), (229, 429), (425, 366), (247, 399), (408, 345)]

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
        bbox_info_dict['frame_num'] = 0                                     # 车辆进入画面帧数
        bbox_info_dict['up_down_speed'] = 0                                 # 车辆速度（上下行）
        bbox_info_dict['left_right_speed'] = 0                              # 车辆速度（左右行）
        bbox_info_dict['alarm_flage'] = 0                                   # 抓拍标志位
        bbox_info_dict['alarm_intersect_point'] = []                        # 抓拍交点坐标
        bbox_info_dict['alarm_capture_line_id'] = 0                         # 抓拍线id
        bbox_info_dict['alarm_car_line'] = []                               # 抓拍车辆线

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
        bbox_state_dict['frame_num'] = 0                                    # 车辆进入画面帧数
        bbox_state_dict['up_down_speed'] = 0                                # 车辆速度（上下行）
        bbox_state_dict['left_right_speed'] = 0                             # 车辆速度（左右行）
        bbox_state_dict['center_point_list'] = []                           # 车辆中心点轨迹（多帧）
        bbox_state_dict['alarm_flage'] = 0                                  # 抓拍标志位
        bbox_state_dict['alarm_intersect_point'] = []                       # 抓拍交点坐标
        bbox_state_dict['alarm_capture_line_id'] = 0                        # 抓拍线id
        bbox_state_dict['alarm_car_line'] = []                              # 抓拍车辆线

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

        # # captute
        capture_points =  self.capture_points

        bbox_info_list = self.update_capture( bbox_info_list )

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
            bbox_info_dict['frame_num'] = 0                                     # 车辆进入画面帧数
            bbox_info_dict['up_down_speed'] = 0                                 # 车辆速度（上下行）
            bbox_info_dict['left_right_speed'] = 0                              # 车辆速度（左右行）
            bbox_info_dict['alarm_flage'] = 0                                   # 抓拍标志位
            bbox_info_dict['alarm_intersect_point'] = []                        # 抓拍交点坐标
            bbox_info_dict['alarm_capture_line_id'] = 0                         # 抓拍线id
            bbox_info_dict['alarm_car_line'] = []                               # 抓拍车辆线

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

                    bbox_info_idx['up_down_state'] = bbox_state_idy['up_down_state']
                    bbox_info_idx['up_down_state_frame_num'] = bbox_state_idy['up_down_state_frame_num']
                    bbox_info_idx['left_right_state'] = bbox_state_idy['left_right_state']
                    bbox_info_idx['left_right_state_frame_num'] = bbox_state_idy['left_right_state_frame_num']
                    bbox_info_idx['stable_loc'] = bbox_state_idy['stable_loc']
                    bbox_info_idx['frame_num'] = bbox_state_idy['frame_num']
                    bbox_info_idx['up_down_speed'] = bbox_state_idy['up_down_speed']
                    bbox_info_idx['left_right_speed'] = bbox_state_idy['left_right_speed']

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
                bbox_state_dict['frame_num'] = 0                                    # 车辆进入画面帧数
                bbox_state_dict['up_down_speed'] = 0                                # 车辆速度（上下行）
                bbox_state_dict['left_right_speed'] = 0                             # 车辆速度（左右行）
                bbox_state_dict['center_point_list'] = []                           # 车辆中心点轨迹（多帧）
                bbox_state_dict['car_disappear_frame_num'] = 0                      # 车辆消失画面帧数
                bbox_state_dict['alarm_flage'] = 0                                  # 抓拍标志位
                bbox_state_dict['alarm_intersect_point'] = []                       # 抓拍交点坐标
                bbox_state_dict['alarm_capture_line_id'] = 0                        # 抓拍线id
                bbox_state_dict['alarm_car_line'] = []                              # 抓拍车辆线

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
    

    def update_capture(self, bbox_info_list):

        # 标定线纵向平行线
        # y = kx + b
        capture_point_top = self.capture_points[17]
        capture_point_bottom = self.capture_points[1]
        capture_rigrt_line_k = float(capture_point_top[1] - capture_point_bottom[1]) / float(capture_point_top[0] - capture_point_bottom[0] + 1e-5);
        capture_right_line_b = float(capture_point_top[1] - capture_point_top[0] * capture_rigrt_line_k);

        capture_point_top = self.capture_points[16]
        capture_point_bottom = self.capture_points[0]
        capture_left_line_k = float(capture_point_top[1] - capture_point_bottom[1]) / float(capture_point_top[0] - capture_point_bottom[0] + 1e-5);
        capture_left_line_b = float(capture_point_top[1] - capture_point_top[0] * capture_left_line_k);

        # 报警逻辑（前向C28）
        # 车辆从右侧驶入 -> 左侧驶出
        for idx in range(len(bbox_info_list)):
            bbox_info_idx = bbox_info_list[idx]

            for _, bbox_state_idy in self.params_dict['bbox_state_container'].items():
                
                # 容器中存在追踪对象
                if bbox_info_idx['id'] == bbox_state_idy['id']:
                    
                    # 车辆坐标
                    car_right_bottom_x = bbox_state_idy['loc'][2] 
                    car_right_bottom_y = bbox_state_idy['loc'][3]
                    car_center_x = int((bbox_state_idy['loc'][0] + bbox_state_idy['loc'][2]) / 2)
                    car_center_y = int((bbox_state_idy['loc'][1] + bbox_state_idy['loc'][3]) / 2)

                    # 1、判断车辆是否位于报警区域内部
                    alarm_flage = False
                    alarm_car_line = []
                    alarm_intersect_point = []
                    for idx in range(int(len(self.capture_points) / 2)):

                        # 标定线
                        point_1 = tuple( self.capture_points[ 2 * idx] )
                        point_2 = tuple( self.capture_points[ 2 * idx + 1] ) 
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
                        alarm_car_line = [car_line_k, car_line_b]
                        
                        # 判断标定线和车辆中心点纵向平行线 是否相交
                        point_intersect_x = (car_line_b - point_line_b)/(point_line_k - car_line_k)
                        point_intersect_y = point_line_k * point_intersect_x + point_line_b
                        alarm_intersect_point = [point_intersect_x, point_intersect_y]

                        if point_intersect_x > min(point_1[0], point_2[0]) and point_intersect_x <= max(point_1[0], point_2[0]) and \
                            point_intersect_y > min(point_1[1], point_2[1]) and point_intersect_y <= max(point_1[1], point_2[1]) and \
                            point_intersect_x > bbox_state_idy['loc'][0] and point_intersect_x <= bbox_state_idy['loc'][2] and \
                            point_intersect_y > bbox_state_idy['loc'][1] and point_intersect_y <= bbox_state_idy['loc'][3]:

                            alarm_flage = True
                            break
                    
                    # 2、输出报警线，根据车辆右下角和报警线之间的关系
                    alarm_capture_line_id = -1
                    if alarm_flage:

                        for idx in range(int(len(self.capture_points) / 2)):
                            # 标定线
                            point_1 = tuple( self.capture_points[ 2 * idx] )
                            point_2 = tuple( self.capture_points[ 2 * idx + 1] ) 
                            point_center_x = int((point_1[0] + point_2[0]) / 2)
                            # y = kx + b
                            point_line_k = float(point_1[1] - point_2[1]) / float(point_1[0] - point_2[0] + 1e-5);
                            point_line_b = float(point_1[1] - point_1[0] * point_line_k);

                            # dist
                            dist = ((car_right_bottom_y - point_line_b) / (point_line_k + 1e-5) - car_right_bottom_x)
                            if dist <= 0:
                                alarm_capture_line_id = idx
                                break

                    bbox_state_idy['alarm_flage'] = alarm_flage
                    bbox_state_idy['alarm_intersect_point'] = alarm_intersect_point
                    bbox_state_idy['alarm_capture_line_id'] = alarm_capture_line_id
                    bbox_state_idy['alarm_car_line'] = alarm_car_line
                    bbox_info_idx['alarm_flage'] = bbox_state_idy['alarm_flage']
                    bbox_info_idx['alarm_intersect_point'] = bbox_state_idy['alarm_intersect_point']
                    bbox_info_idx['alarm_capture_line_id'] = bbox_state_idy['alarm_capture_line_id']
                    bbox_info_idx['alarm_car_line'] = bbox_state_idy['alarm_car_line']

        return bbox_info_list