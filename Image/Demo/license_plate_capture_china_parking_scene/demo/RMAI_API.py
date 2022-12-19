from collections import Counter
import cv2
import numpy as np
import sys 
import random

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.detection2d.ssd_rfb_crossdatatraining.test_tools import SSDDetector
from Image.recognition2d.license_plate_ocr.infer.lpr import LPR
from Image.Demo.license_plate_capture_china_vehicle_scene.sort.mot_sort import Sort


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

        # self.image_width = 2592
        # self.image_height = 1920
        self.image_width = 1280
        self.image_height = 720
        # self.image_width = 640
        # self.image_height = 368

        # 2022-09-08-00 zt_crop
        # pytorch 
        self.ssd_car_plate_prototxt = None
        self.ssd_car_plate_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-09-08-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_zf_zt_crop_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_225.pth"
        # # caffe
        # self.ssd_car_plate_prototxt = ""
        # self.ssd_car_plate_model_path = ""
        # openvino
        # self.ssd_car_plate_prototxt = None
        # self.ssd_car_plate_model_path = ""

        self.ssd_caffe_bool = False
        self.ssd_openvino_bool = False

        # 是否将 car\bus\truck 合并为一类输出
        self.merge_class_bool = True
        self.merge_class_name = 'car_bus_truck'
        self.license_plate_name = 'license_plate'

        # sort
        self.max_age = 10
        self.min_hits = 3 
        self.iou_threshold = 0.3

        # lincense plate reader
        self.lpr_type = "china"

        # # china: lpr_lxn
        # self.lpr_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr_lxn/china_softmax.prototxt"
        # self.lpr_caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr_lxn/china.caffemodel"
        # china: lpr_zg
        self.lpr_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr_zg/china/0628/china_double_softmax.prototxt"
        self.lpr_caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr_zg/china/0628/china_double.caffemodel"
        self.lpr_prefix_beam_search_bool = False

        # 缓存间隔
        self.cache_interval = 2
        # 缓存容器长度
        self.cache_container_length = 8

        # 状态容器长度
        self.bbox_state_container_length = 10       # 车牌框连续丢失上报，从容器中清除该车辆信息

        # 更新车辆行驶状态
        self.update_state_num_threshold = 5         # 行驶状态计数最大值，用于记录车辆处于同一行驶状态的帧数
        self.update_state_threshold = 1
        self.update_state_stable_loc_alpha = float(0.6)   # 平滑检测框参数

        # 报警时间长短
        self.capture_frame_num_threshold = 16

        # 是否通过 roi 区域屏蔽部分检测结果
        self.roi_bool = False
        # self.roi_bool = True
        self.roi_area = [0, 0, self.image_width, self.image_height]

        # 车牌长宽阈值
        # ignore：
        self.plate_height = [50, 1000]
        self.plate_width = [50, 1000]

        # 抓拍线
        self.capture_line_up_down_ratio = [0.1, 0.9]
        self.capture_line_left_right_ratio = [0.1, 0.9]
        self.capture_plate_frame_threshold = 5
        self.capture_outtime_frame_threshold_01 = 50
        self.capture_outtime_frame_threshold_02 = 250
        self.capture_plate_up_down_distance_boundary_threshold = 50
        self.capture_plate_left_right_distance_boundary_threshold = 50
        self.capture_plate_ocr_score_threshold = 0.8
        self.capture_plate_ocr_frame_threshold = 4
        self.capture_times_threshold = 7

        # 上下限阈值 & 左右限阈值
        if self.roi_bool:
            self.Up_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_up_down_ratio[0]
            self.Down_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_up_down_ratio[1]
            self.Left_threshold = self.roi_area[0] + ( self.roi_area[2] - self.roi_area[0] ) * self.capture_line_left_right_ratio[0]
            self.Right_threshold = self.roi_area[0] + ( self.roi_area[2] - self.roi_area[0] ) * self.capture_line_left_right_ratio[1]
        else:
            self.Up_threshold = self.image_height * self.capture_line_up_down_ratio[0]
            self.Down_threshold = self.image_height * self.capture_line_up_down_ratio[1]
            self.Left_threshold = self.image_width * self.capture_line_left_right_ratio[0]
            self.Right_threshold = self.image_width * self.capture_line_left_right_ratio[1]


    def param_init(self):
        self.params_dict = {}

        # bbox_info_dict
        bbox_info_dict = {}
        bbox_info_dict['id'] = 0                                            # 追踪id
        bbox_info_dict['loc'] = []                                          # 车牌坐标
        bbox_info_dict['stable_loc'] = []                                   # 车牌坐标（稳定）
        bbox_info_dict['up_down_state'] = 'Stop'                            # 车牌状态（上下行）
        bbox_info_dict['up_down_state_frame_num'] = 0                       # 车牌状态（上下行）帧数
        bbox_info_dict['left_right_state'] = 'Stop'                         # 车牌状态（左右行）
        bbox_info_dict['left_right_state_frame_num'] = 0                    # 车牌状态（左右行）帧数
        bbox_info_dict['frame_num'] = 0                                     # 车牌进入画面帧数
        bbox_info_dict['up_down_speed'] = 0                                 # 车牌速度（上下行）
        bbox_info_dict['left_right_speed'] = 0                              # 车牌速度（左右行）
        bbox_info_dict['plate_ocr_num'] = 0                                 # 车牌识别帧数
        bbox_info_dict['plate_ocr'] = ''                                    # 车牌识别结果（单帧）
        bbox_info_dict['plate_ocr_score'] = 0.0                             # 车牌识别结果得分（单帧）
        
        self.params_dict['cache_container'] = []                            # 缓存容器（imag & bbox_info_dict）

        # bbox_state_dict
        bbox_state_dict = {}
        bbox_state_dict['id'] = 0                                           # 追踪id
        bbox_state_dict['loc'] = []                                         # 车牌坐标
        bbox_state_dict['stable_loc'] = []                                  # 车牌坐标（稳定）
        bbox_state_dict['up_down_state'] = 'Stop'                           # 车牌状态（上下行）
        bbox_state_dict['up_down_state_frame_num'] = 0                      # 车牌状态（上下行）帧数
        bbox_state_dict['left_right_state'] = 'Stop'                        # 车牌状态（左右行）
        bbox_state_dict['left_right_state_frame_num'] = 0                   # 车牌状态（左右行）帧数
        bbox_state_dict['frame_num'] = 0                                    # 车牌进入画面帧数
        bbox_state_dict['disappear_frame_num'] = 0                          # 车牌消失画面帧数
        bbox_state_dict['up_down_speed'] = 0                                # 车牌速度（上下行）
        bbox_state_dict['left_right_speed'] = 0                             # 车牌速度（左右行）
        bbox_state_dict['center_point_list'] = []                           # 车牌中心点轨迹（多帧）
        bbox_state_dict['plate_ocr_num'] = 0                                # 车牌识别帧数
        bbox_state_dict['plate_ocr_list'] = []                              # 车牌识别结果（多帧）
        bbox_state_dict['plate_ocr_score_list'] = []                        # 车牌识别结果得分（多帧）
        bbox_state_dict['stop_report_flage'] = False                        # 抓拍标志位
        bbox_state_dict['far_report_flage'] = False                         # 抓拍标志位
        bbox_state_dict['near_report_flage'] = False                        # 抓拍标志位
        bbox_state_dict['left_report_flage'] = False                        # 抓拍标志位
        bbox_state_dict['right_report_flage'] = False                       # 抓拍标志位
        bbox_state_dict['outtime_flage_01'] = False                         # 抓拍标志位
        bbox_state_dict['outtime_flage_02'] = False                         # 抓拍标志位
        bbox_state_dict['report_times'] = 0                                 # 抓拍次数
        
        self.params_dict['bbox_state_container'] = {}                       # 状态信息容器（key: 追踪id, value: bbox_state_dict）

        # capture_dict
        capture_dict = {}                                                   # 抓怕
        capture_dict['id'] = 0                                              # 抓怕id
        capture_dict['flage'] = ''                                          # 抓拍标志信息
        capture_dict['capture_frame_num'] = 0                               # 抓拍帧数
        capture_dict['capture_bool'] = False                                # 抓拍成功标志

        self.params_dict['capture_dict'] = {}                               # 抓拍序列
        self.params_dict['capture_list'] = []                               # 抓拍结果


    def model_init(self):
        # detector
        self.detector = SSDDetector(prototxt=self.ssd_car_plate_prototxt, model_path=self.ssd_car_plate_model_path, ssd_caffe_bool=self.ssd_caffe_bool, ssd_openvino_bool=self.ssd_openvino_bool, merge_class_bool=self.merge_class_bool)

        # tracker
        self.mot_tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_threshold)

        # lincense plate reader
        self.lpr = LPR(self.lpr_caffe_prototxt, self.lpr_caffe_model_path, self.lpr_prefix_beam_search_bool)


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

        # store
        # 跳帧存储原图和检测识别结果
        self.update_cache_container( img, frame_idx, bbox_info_list )

        # 更新状态容器，同时更新车辆行驶状态和帧率
        bbox_info_list = self.update_bbox_state_container( bbox_info_list )
        bbox_state_map = self.params_dict['bbox_state_container']

        # captute
        ## capture_line
        if self.roi_bool:
            capture_line_up_down = [ self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * ratio for ratio in self.capture_line_up_down_ratio ]
            capture_line_left_right = [ self.roi_area[0] + ( self.roi_area[2] - self.roi_area[0] ) * ratio for ratio in self.capture_line_left_right_ratio ]
        else:
            capture_line_up_down = [ self.image_height * ratio for ratio in self.capture_line_up_down_ratio ]
            capture_line_left_right = [ self.image_width * ratio for ratio in self.capture_line_left_right_ratio ]

        self.update_capture_dict()
        capture_dict = self.params_dict['capture_dict']
        capture_res_list = self.update_capture_state( )

        return tracker_bboxes, bbox_info_list, bbox_state_map, capture_line_up_down, capture_line_left_right, capture_dict, capture_res_list


    def update_tracker_bboxes(self, bboxes):

        # tracker
        dets = np.empty((0, 5))
        if self.license_plate_name in bboxes:
            dets = np.concatenate((dets, np.array(bboxes[self.license_plate_name])), axis=0)
                
        tracker_bboxes = self.mot_tracker.update(dets)

        return tracker_bboxes


    def update_bbox_info(self, img, bboxes, tracker_bboxes):

        # gray img for lincense plate reader
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bbox_info_list = []
        for idx in range(len(tracker_bboxes)):
            # init 
            # bbox_info_dict
            bbox_info_dict = {}
            bbox_info_dict['id'] = 0                                            # 追踪id
            bbox_info_dict['loc'] = []                                          # 车牌坐标
            bbox_info_dict['stable_loc'] = []                                   # 车牌坐标（稳定）
            bbox_info_dict['up_down_state'] = 'Stop'                            # 车牌状态（上下行）
            bbox_info_dict['up_down_state_frame_num'] = 0                       # 车牌状态（上下行）帧数
            bbox_info_dict['left_right_state'] = 'Stop'                         # 车牌状态（左右行）
            bbox_info_dict['left_right_state_frame_num'] = 0                    # 车牌状态（左右行）帧数
            bbox_info_dict['frame_num'] = 0                                     # 车牌进入画面帧数
            bbox_info_dict['up_down_speed'] = 0                                 # 车牌速度（上下行）
            bbox_info_dict['left_right_speed'] = 0                              # 车牌速度（左右行）
            bbox_info_dict['plate_ocr_num'] = 0                                 # 车牌识别帧数
            bbox_info_dict['plate_ocr'] = ''                                    # 车牌识别结果（单帧）
            bbox_info_dict['plate_ocr_score'] = 0.0                             # 车牌识别结果得分（单帧）

            # license plate
            tracker_bbox = tracker_bboxes[idx]
            bbox_info_dict['id'] = tracker_bbox[-1]
            bbox_info_dict['loc'] = tracker_bbox[0:4]
                        
            # 按照车牌宽高过滤车牌
            if (bbox_info_dict['loc'][3] - bbox_info_dict['loc'][1] > self.plate_height[0]) and \
                (bbox_info_dict['loc'][3] - bbox_info_dict['loc'][1] < self.plate_height[1]) and \
                (bbox_info_dict['loc'][2] - bbox_info_dict['loc'][0] > self.plate_width[0]) and \
                (bbox_info_dict['loc'][2] - bbox_info_dict['loc'][0] < self.plate_width[1]):
                
                # lincense plate reader
                # crop
                x1 = min(max(0, int(bbox_info_dict['loc'][0])), self.image_width)
                x2 = min(max(0, int(bbox_info_dict['loc'][2])), self.image_width)
                y1 = min(max(0, int(bbox_info_dict['loc'][1])), self.image_height)
                y2 = min(max(0, int(bbox_info_dict['loc'][3])), self.image_height)
                crop_img = gray_img[y1: y2, x1: x2]

                plate_ocr, plate_scors_list = self.lpr.run(crop_img)
                
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
        pop_key_list = []
        for key, bbox_state_idy in self.params_dict['bbox_state_container'].items():
          
            # pop
            if bbox_state_idy['disappear_frame_num'] > self.bbox_state_container_length:
                pop_key_list.append(key)
            
            bbox_state_idy['disappear_frame_num'] += 1
        
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

                    # 更新车牌速度
                    new_stable_loc = self.update_state_stable_loc_alpha * bbox_state_idy['stable_loc'] +  (1 - self.update_state_stable_loc_alpha) * bbox_info_idx['loc']
                    old_stable_center_x = ( bbox_state_idy['stable_loc'][0] + bbox_state_idy['stable_loc'][2] ) / 2.0
                    new_stable_center_x = ( new_stable_loc[0] + new_stable_loc[2] ) / 2.0
                    old_stable_center_y = ( bbox_state_idy['stable_loc'][1] + bbox_state_idy['stable_loc'][3] ) / 2.0
                    new_stable_center_y = ( new_stable_loc[1] + new_stable_loc[3] ) / 2.0
                    bbox_state_idy['up_down_speed'] = (old_stable_center_y - new_stable_center_y) / float(bbox_state_idy['disappear_frame_num'])
                    bbox_state_idy['left_right_speed'] = (old_stable_center_x - new_stable_center_x) / float(bbox_state_idy['disappear_frame_num'])
                    bbox_state_idy['stable_loc'] = new_stable_loc

                    bbox_state_idy['disappear_frame_num'] = 0

                    # 车牌状态判断（上下行）
                    if bbox_state_idy['up_down_speed'] > self.update_state_threshold:
                        bbox_state = 'Far'
                    elif bbox_state_idy['up_down_speed'] < ( -1 * self.update_state_threshold ):
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
                    
                    # 车牌状态判断（左右行）
                    if bbox_state_idy['left_right_speed'] > self.update_state_threshold:
                        bbox_state = 'Left'
                    elif bbox_state_idy['left_right_speed'] < ( -1 * self.update_state_threshold ):
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

                    # 更新车牌识别有效帧数
                    loc_center_x = (bbox_state_idy['loc'][0] + bbox_state_idy['loc'][2]) / 2.0
                    loc_center_y = (bbox_state_idy['loc'][1] + bbox_state_idy['loc'][3]) / 2.0

                    if not bbox_info_idx['plate_ocr'] == '':
                        if loc_center_x > self.Left_threshold and loc_center_x < self.Right_threshold and \
                            loc_center_y > self.Up_threshold and loc_center_y < self.Down_threshold:
                            bbox_state_idy['plate_ocr_num'] += 1
                            bbox_state_idy['plate_ocr_list'].append(bbox_info_idx['plate_ocr'])
                            bbox_state_idy['plate_ocr_score_list'].append(bbox_info_idx['plate_ocr_score'])

                    bbox_info_idx['stable_loc'] = bbox_state_idy['stable_loc']
                    bbox_info_idx['up_down_state'] = bbox_state_idy['up_down_state']
                    bbox_info_idx['up_down_state_frame_num'] = bbox_state_idy['up_down_state_frame_num']
                    bbox_info_idx['left_right_state'] = bbox_state_idy['left_right_state']
                    bbox_info_idx['left_right_state_frame_num'] = bbox_state_idy['left_right_state_frame_num']
                    bbox_info_idx['frame_num'] = bbox_state_idy['frame_num']
                    bbox_info_idx['plate_ocr_num'] = bbox_state_idy['plate_ocr_num']
                    bbox_info_idx['up_down_speed'] = bbox_state_idy['up_down_speed']
                    bbox_info_idx['left_right_speed'] = bbox_state_idy['left_right_speed']

            if is_new_id_bool:

                # bbox_state_dict
                bbox_state_dict = {}
                bbox_state_dict['id'] = 0                                           # 追踪id
                bbox_state_dict['loc'] = []                                         # 车牌坐标
                bbox_state_dict['stable_loc'] = []                                  # 车牌坐标（稳定）
                bbox_state_dict['up_down_state'] = 'Stop'                           # 车牌状态（上下行）
                bbox_state_dict['up_down_state_frame_num'] = 0                      # 车牌状态（上下行）帧数
                bbox_state_dict['left_right_state'] = 'Stop'                        # 车牌状态（左右行）
                bbox_state_dict['left_right_state_frame_num'] = 0                   # 车牌状态（左右行）帧数
                bbox_state_dict['frame_num'] = 0                                    # 车牌进入画面帧数
                bbox_state_dict['disappear_frame_num'] = 0                          # 车牌消失画面帧数
                bbox_state_dict['up_down_speed'] = 0                                # 车牌速度（上下行）
                bbox_state_dict['left_right_speed'] = 0                             # 车牌速度（左右行）
                bbox_state_dict['center_point_list'] = []                           # 车牌中心点轨迹（多帧）
                bbox_state_dict['plate_ocr_num'] = 0                                # 车牌识别帧数
                bbox_state_dict['plate_ocr_list'] = []                              # 车牌识别结果（多帧）
                bbox_state_dict['plate_ocr_score_list'] = []                        # 车牌识别结果得分（多帧）
                bbox_state_dict['stop_report_flage'] = False                        # 抓拍标志位
                bbox_state_dict['far_report_flage'] = False                         # 抓拍标志位
                bbox_state_dict['near_report_flage'] = False                        # 抓拍标志位
                bbox_state_dict['left_report_flage'] = False                        # 抓拍标志位
                bbox_state_dict['right_report_flage'] = False                       # 抓拍标志位
                bbox_state_dict['outtime_flage_01'] = False                         # 抓拍标志位
                bbox_state_dict['outtime_flage_02'] = False                         # 抓拍标志位
                bbox_state_dict['report_times'] = 0                                 # 抓拍次数

                bbox_state_dict['id'] = bbox_info_idx['id']
                bbox_state_dict['loc'] = bbox_info_idx['loc']
                bbox_state_dict['stable_loc'] = bbox_info_idx['loc']

                bbox_info_idx['stable_loc'] = bbox_info_idx['loc']

                # 更新车牌识别有效帧数
                loc_center_x = (bbox_info_idx['loc'][0] + bbox_info_idx['loc'][2]) / 2.0
                loc_center_y = (bbox_info_idx['loc'][1] + bbox_info_idx['loc'][3]) / 2.0
                
                if not bbox_info_idx['plate_ocr'] == '':
                    if loc_center_x > self.Left_threshold and loc_center_x < self.Right_threshold and \
                        loc_center_y > self.Up_threshold and loc_center_y < self.Down_threshold:
                        bbox_state_dict['plate_ocr_num'] += 1
                        bbox_state_dict['plate_ocr_list'].append(bbox_info_idx['plate_ocr'])
                        bbox_state_dict['plate_ocr_score_list'].append(bbox_info_idx['plate_ocr_score'])

                self.params_dict['bbox_state_container'][bbox_state_dict['id']] = bbox_state_dict

        # update
        # center_point_list
        pop_key_list = []
        for key, bbox_state_idy in self.params_dict['bbox_state_container'].items():

            bbox_state_idy['center_point_list'].append( ( (bbox_state_idy['loc'][0] + bbox_state_idy['loc'][2]) / 2 , (bbox_state_idy['loc'][1] + bbox_state_idy['loc'][3]) / 2 ) )
            
            if len( bbox_state_idy['center_point_list'] ) > self.bbox_state_container_length:
                bbox_state_idy['center_point_list'].pop(0)

        return bbox_info_list


    def update_capture_dict(self):
        
        # update
        pop_key_list = []
        for key, capture_dict_idy in self.params_dict['capture_dict'].items():
          
            # pop
            if capture_dict_idy['capture_frame_num'] > self.capture_frame_num_threshold:
                pop_key_list.append(key)
            
            elif capture_dict_idy['capture_bool']:
                pop_key_list.append(key)
            
            capture_dict_idy['capture_frame_num'] += 1
        
        # pop
        for idx in range(len(pop_key_list)):
            self.params_dict['capture_dict'].pop(pop_key_list[idx])
       
        # 报警逻辑
        # 1、快速抓怕：车牌行驶到触发线，执行抓拍。（近处、远处、左边、右边）
        # 2、稳定抓拍：车牌位于图像正中央，超过一定时间，执行抓拍。
        for _, bbox_state_idy in self.params_dict['bbox_state_container'].items():
            # init 
            stop_flage = False
            near_flage = False
            far_flage = False
            left_flage = False
            right_flage = False
            outtime_flage_01 = False
            outtime_flage_02 = False
            report_flage = False

            loc_center_x = (bbox_state_idy['loc'][0] + bbox_state_idy['loc'][2]) / 2.0
            loc_center_y = (bbox_state_idy['loc'][1] + bbox_state_idy['loc'][3]) / 2.0
            
            # 如果车辆向近处行驶, bbox_state_idy['up_down_state_frame_num'] >= 3 条件用于避免刚进 ROI 或者车辆静止状态下的误判
            if bbox_state_idy['up_down_state'] == 'Near' and bbox_state_idy['up_down_state_frame_num'] >= 3:
                if abs(loc_center_y - self.Down_threshold) < self.capture_plate_up_down_distance_boundary_threshold and \
                    bbox_state_idy['plate_ocr_num'] > self.capture_plate_frame_threshold:
                    near_flage = True

            # 如果车辆向远处行驶，bbox_state_idy['up_down_state_frame_num'] 条件用于避免刚进 ROI 或者车辆静止状态下的误判
            if bbox_state_idy['up_down_state'] == 'Far' and bbox_state_idy['up_down_state_frame_num'] >= 3:
                if abs(loc_center_y - self.Up_threshold) < self.capture_plate_up_down_distance_boundary_threshold and \
                    bbox_state_idy['plate_ocr_num'] > self.capture_plate_frame_threshold:
                    far_flage = True

            # 如果车辆向左边行驶
            if bbox_state_idy['left_right_state'] == 'Left' and bbox_state_idy['left_right_state_frame_num'] >= 3:
                if abs(loc_center_x - self.Left_threshold) < self.capture_plate_left_right_distance_boundary_threshold and \
                    bbox_state_idy['plate_ocr_num'] > self.capture_plate_frame_threshold:
                    left_flage = True
            
            # 如果车辆向右边行驶
            if bbox_state_idy['left_right_state'] == 'Right' and bbox_state_idy['left_right_state_frame_num'] >= 3:
                if abs(loc_center_x - self.Right_threshold) < self.capture_plate_left_right_distance_boundary_threshold and \
                    bbox_state_idy['plate_ocr_num'] > self.capture_plate_frame_threshold:
                    right_flage = True

            # 如果车辆相对静止，直接上报：
            if ( bbox_state_idy['up_down_state'] == 'Stop' and bbox_state_idy['up_down_state_frame_num'] >= 3 ) and \
                ( bbox_state_idy['left_right_state'] == 'Stop' and bbox_state_idy['left_right_state_frame_num'] >= 3 ):
                if abs(loc_center_y - self.Down_threshold) > self.capture_plate_up_down_distance_boundary_threshold and \
                        abs(loc_center_y - self.Up_threshold) > self.capture_plate_up_down_distance_boundary_threshold and \
                        abs(loc_center_x - self.Left_threshold) > self.capture_plate_left_right_distance_boundary_threshold and \
                        abs(loc_center_x - self.Right_threshold) > self.capture_plate_left_right_distance_boundary_threshold and \
                        bbox_state_idy['disappear_frame_num'] == 0 and bbox_state_idy['plate_ocr_num'] > self.capture_plate_frame_threshold:
                    stop_flage = True

            # 如果车辆在视野内，超过 50 帧
            if bbox_state_idy['plate_ocr_num'] > self.capture_outtime_frame_threshold_01:
                outtime_flage_01 = True

            # 如果车辆在视野内，超过 150 帧
            if bbox_state_idy['plate_ocr_num'] > self.capture_outtime_frame_threshold_02:
                outtime_flage_02 = True
                         
            # 更新 capture_dict 抓拍字典
            capture_dict = {}                                                   # 抓拍
            capture_dict['id'] = bbox_state_idy['id']                           # 抓拍id
            capture_dict['flage'] = ''                                          # 抓拍标志信息
            capture_dict['capture_frame_num'] = 0                               # 抓拍帧数
            capture_dict['capture_bool'] = False                                # 抓拍成功标志

            if stop_flage and not bbox_state_idy['stop_report_flage']:
                bbox_state_idy['stop_report_flage'] = True
                report_flage = True
                capture_dict['flage'] = 'stop_flage'
            
            if near_flage and not bbox_state_idy['near_report_flage']:
                bbox_state_idy['near_report_flage'] = True
                report_flage = True
                capture_dict['flage'] = 'near_flage'

            if far_flage and not bbox_state_idy['far_report_flage']:
                bbox_state_idy['far_report_flage'] = True
                report_flage = True
                capture_dict['flage'] = 'far_flage'
                
            if left_flage and not bbox_state_idy['left_report_flage']:
                bbox_state_idy['left_report_flage'] = True
                report_flage = True
                capture_dict['flage'] = 'left_flage'

            if right_flage and not bbox_state_idy['right_report_flage']:
                bbox_state_idy['right_report_flage'] = True
                report_flage = True
                capture_dict['flage'] = 'right_flage'

            if outtime_flage_01 and not bbox_state_idy['outtime_flage_01']:
                bbox_state_idy['outtime_flage_01'] = True
                report_flage = True
                capture_dict['flage'] = 'outtime_flage_01'

            if outtime_flage_02 and not bbox_state_idy['outtime_flage_02']:
                bbox_state_idy['outtime_flage_02'] = True
                report_flage = True
                capture_dict['flage'] = 'outtime_flage_02'

            if report_flage and bbox_state_idy['report_times'] < self.capture_times_threshold:
                
                bbox_state_idy['report_times'] += 1;

                # 更新 capture_dict 抓拍字典
                if bbox_state_idy['id'] not in self.params_dict['capture_dict']:

                    self.params_dict['capture_dict'][capture_dict['id']] = capture_dict                 # 抓拍序列
            
        return


    def update_capture_state(self):
        
        capture_res_list = []
        for capture_id_idx, capture_dict_idy in self.params_dict['capture_dict'].items():
            
            capture_flage_idx = capture_dict_idy['flage']

            # init 
            capture_res_dict = {}
            capture_res_dict['id'] = capture_id_idx
            capture_res_dict['plate_ocr'] = ''
            capture_res_dict['plate_state'] = ''
            capture_res_dict['img_bbox_info'] = []

            for _, bbox_state_idy in self.params_dict['bbox_state_container'].items():

                if bbox_state_idy['id'] == capture_id_idx:

                    # capture_flage
                    plate_ocr_np = np.array(bbox_state_idy['plate_ocr_list'])
                    plate_ocr_score_np = np.array(bbox_state_idy['plate_ocr_score_list'])

                    # 获得抓拍序列
                    if len(plate_ocr_np[plate_ocr_score_np > self.capture_plate_ocr_score_threshold]):
                        capture_license_palte, capture_license_palte_frame = Counter(list(plate_ocr_np[plate_ocr_score_np > self.capture_plate_ocr_score_threshold])).most_common(1)[0]

                        if capture_license_palte_frame >= self.capture_plate_ocr_frame_threshold:
                            capture_from_container_list = self.find_capture_plate(bbox_state_idy['id'], capture_license_palte)

                            # 抓拍车牌
                            if len(capture_from_container_list):
                                capture_dict_idy['capture_bool'] = True
                                capture_res_dict['plate_ocr'] = capture_license_palte
                                capture_res_dict['img_bbox_info'] = capture_from_container_list

                                # 抓到车牌，标志位置1
                                if capture_flage_idx == 'stop_flage':
                                    capture_res_dict['plate_state'] = 'stop'

                                # 抓到车牌，标志位置1
                                if capture_flage_idx == 'near_flage':
                                    capture_res_dict['plate_state'] = 'near'

                                # 抓到车牌，标志位置1
                                if capture_flage_idx == 'far_flage':
                                    capture_res_dict['plate_state'] = 'far'

                                # 抓到车牌，标志位置1
                                if capture_flage_idx == 'left_flage':
                                    capture_res_dict['plate_state'] = 'left'

                                # 抓到车牌，标志位置1
                                if capture_flage_idx == 'right_flage':
                                    capture_res_dict['plate_state'] = 'right'

                                # 抓到车牌，标志位置1
                                if capture_flage_idx == 'outtime_flage_01':
                                    capture_res_dict['plate_state'] = 'outtime_01'

                                # 抓到车牌，标志位置1
                                if capture_flage_idx == 'outtime_flage_02':
                                    capture_res_dict['plate_state'] = 'outtime_02'
                                
                                if capture_res_dict['plate_ocr'] not in self.params_dict['capture_list']:

                                    self.params_dict['capture_list'].append(capture_res_dict['plate_ocr'])
                                    capture_res_list.append(capture_res_dict)

        return capture_res_list


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