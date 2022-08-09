from collections import Counter
import numpy as np
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo')
# sys.path.insert(0, '/yuanhuan/code/demo')
# sys.path.insert(0, 'E:\\project\\demo')
from Image.detection2d.ssd_rfb_crossdatatraining.test_tools import SSDDetector
from Image.Basic.video_capture.plate_color.plate_color import update_plate_color
from Image.Demo.license_plate_capture_gantry_scene.sort.mot_sort import Sort


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

        self.image_width = 2592
        self.image_height = 1920

        # detector
        # # 2022-07-30-00
        # # pytorch 
        self.ssd_car_plate_prototxt = None
        self.ssd_car_plate_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-07-22-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
        # # caffe
        # self.ssd_car_plate_prototxt = "/mnt/huanyuan/model_final/image_model/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-07-22-00/FPN_RFB_3class_3attri_noDilation_prior.prototxt"
        # self.ssd_car_plate_model_path = "/mnt/huanyuan/model_final/image_model/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-07-22-00/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-07-22-00.caffemodel"
        # openvino
        # self.ssd_car_plate_prototxt = None
        # self.ssd_car_plate_model_path = "/mnt/huanyuan/model_final/image_model/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-07-22-00/openvino_model/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-07-22-00.xml"

        # self.ssd_car_plate_prototxt = self.model_prototxt
        # self.ssd_car_plate_model_path = self.model_path

        self.ssd_caffe_bool = False
        self.ssd_openvino_bool = False
        self.gpu_bool = self.GPU

        # 是否将 car\bus\truck 合并为一类输出
        self.merge_class_bool = None
        self.merge_class_name = 'car_bus_truck'
        self.car_attri_name_list = [ 'car', 'bus', 'truck' ]
        self.license_plate_name = 'license_plate'

        # sort
        self.max_age = 10
        self.min_hits = 3 
        self.iou_threshold = 0.3

        # 状态容器长度
        self.bbox_state_container_length = 10       # 车辆框连续丢失上报，从容器中清除该车辆信息

        # 更新车辆行驶状态
        self.update_state_container_length = 1      # 车辆框坐标容器大小，用于判断车辆状态
        self.update_state_num_threshold = 5         # 车辆行驶状态计数最大值，用于记录车辆处于同一行驶状态的帧数
        self.update_state_threshold = 1
        self.update_state_stable_loc_alpha = float(0.6)   # 平滑车辆框参数

        # 报警时间长短
        self.capture_frame_num_threshold = 16

        # 是否通过 roi 区域屏蔽部分检测结果
        # self.roi_bool = False
        self.roi_bool = True
        # 2M：16:9
        # args.roi_area = [270, 270, 1650, 1080]
        # # 5M：16:9
        self.roi_area = [0, 462, 2592, 1920]

        # 抓拍线
        self.capture_line_ratio = [0.2, 0.8]
        self.capture_plate_up_down_distance_boundary_threshold = 50
        self.capture_plate_left_right_distance_boundary_threshold = 50


    def param_init(self):
        self.params_dict = {}

        # bbox_info_dict
        bbox_info_dict = {}
        bbox_info_dict['id'] = 0                                            # 追踪id
        bbox_info_dict['loc'] = []                                          # 车辆坐标
        bbox_info_dict['attri'] = 'None'                                    # 车辆属性：car, bus, truck
        bbox_info_dict['stable_loc'] = []                                   # 车辆坐标（稳定）
        bbox_info_dict['up_down_state'] = 'Stop'                            # 车辆状态（上下行）
        bbox_info_dict['up_down_state_frame_num'] = 0                       # 车辆状态（上下行）帧数
        bbox_info_dict['left_right_state'] = 'Stop'                         # 车辆状态（左右行）
        bbox_info_dict['left_right_state_frame_num'] = 0                    # 车辆状态（左右行）帧数
        bbox_info_dict['frame_num'] = 0                                     # 车辆进入画面帧数
        bbox_info_dict['up_down_speed'] = 0                                 # 车辆速度（上下行）
        bbox_info_dict['left_right_speed'] = 0                              # 车辆速度（左右行）
        bbox_info_dict['plate_loc'] = []                                    # 车牌坐标
        bbox_info_dict['plate_color'] = 'None'                              # 车牌颜色

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
        bbox_state_dict['start_frame'] = 0                                  # 车辆驶入画面的帧数
        bbox_state_dict['end_frame'] = 0                                    # 车辆驶出画面的帧数
        bbox_state_dict['attri_list'] = []                                  # 车辆属性列表
        bbox_state_dict['plate_color_list'] = []                            # 车牌颜色列表
        bbox_state_dict['car_disappear_frame_num'] = 0                      # 车辆消失画面帧数
        bbox_state_dict['up_report_flage'] = False                          # 上行抓拍标志位
        bbox_state_dict['down_report_flage'] = False                        # 下行抓拍标志位
        bbox_state_dict['left_report_flage'] = False                        # 左行抓拍标志位
        bbox_state_dict['right_report_flage'] = False                       # 右行抓拍标志位

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
        self.detector = SSDDetector(prototxt=self.ssd_car_plate_prototxt, model_path=self.ssd_car_plate_model_path, ssd_caffe_bool=self.ssd_caffe_bool, ssd_openvino_bool=self.ssd_openvino_bool, merge_class_bool=self.merge_class_bool, gpu_bool=self.gpu_bool)

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
        bbox_info_list = self.update_bbox_state_container( bbox_info_list, frame_idx )
        box_state_map = self.params_dict['bbox_state_container']

        # captute
        self.update_capture_dict()
        capture_dict = self.params_dict['capture_dict']
        capture_result = self.update_capture_state( frame_idx )

        return bbox_info_list, capture_dict, capture_result


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

            # # 方案一：使用 IOU 判断
            # intersect_iou = intersect(car_roi, match_roi_idx)

            # # 计算车牌检测框与车辆检测框的交集区域，大于 0.0 则认为该车牌属于该车辆
            # if intersect_iou > 0.0:
            #     # 默认车牌均是在车辆的下沿
            #     if (car_roi[1] + car_roi[3] / 2.0) < (match_roi_idx[1] + match_roi_idx[3] / 2.0):
            #         matched_roi_list.append(license_plate_list[idx])

            # 方案二：计算车牌框完全位于车框内
            bool_in = bool_box_in_roi(match_roi_idx, car_roi)
            if bool_in:
                # 默认车牌均是在车辆的下沿
                if (car_roi[1] + car_roi[3] / 2.0) < (match_roi_idx[1] + match_roi_idx[3] / 2.0):
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
            bbox_info_dict['loc'] = []                                          # 车辆坐标
            bbox_info_dict['attri'] = 'None'                                    # 车辆属性：car, bus, truck
            bbox_info_dict['stable_loc'] = []                                   # 车辆坐标（稳定）
            bbox_info_dict['up_down_state'] = 'Stop'                            # 车辆状态（上下行）
            bbox_info_dict['up_down_state_frame_num'] = 0                       # 车辆状态（上下行）帧数
            bbox_info_dict['left_right_state'] = 'Stop'                         # 车辆状态（左右行）
            bbox_info_dict['left_right_state_frame_num'] = 0                    # 车辆状态（左右行）帧数
            bbox_info_dict['frame_num'] = 0                                     # 车辆进入画面帧数
            bbox_info_dict['up_down_speed'] = 0                                 # 车辆速度（上下行）
            bbox_info_dict['left_right_speed'] = 0                              # 车辆速度（左右行）
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
                match_car_roi = self.match_bbox_iou(bbox_info_dict['loc'], car_bbox_list)
                if len(match_car_roi):
                    bbox_info_dict['attri'] = match_car_roi[0][-1]

            # license plate
            if self.license_plate_name in bboxes:
                license_plate_roi_list = bboxes[self.license_plate_name]
                # 求相交同时置信度最高的车牌框
                match_license_plate_roi = self.match_car_license_plate(bbox_info_dict['loc'], license_plate_roi_list)

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

                    # 车辆状态判断（上下行）
                    if bbox_state_idy['up_down_speed'] > self.update_state_threshold:
                        bbox_state = 'Up'
                    elif bbox_state_idy['up_down_speed'] < ( -1 * self.update_state_threshold ):
                        bbox_state = 'Down'
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

                    # 更新车牌属性
                    bbox_state_idy['attri_list'].append(bbox_info_idx['attri'])

                    car_bottom_y = bbox_info_idx['loc'][3]
                    # 更新车牌颜色结果
                    if not bbox_info_idx['plate_loc'] == '':
                        if self.roi_bool:
                            # 进入 roi 区域，在记录车牌信息
                            # 原因：远处车牌太小，结果不可信
                            if car_bottom_y > self.roi_area[1]:
                                bbox_state_idy['plate_color_list'].append(bbox_info_idx['plate_color'])
                        else:
                            bbox_state_idy['plate_color_list'].append(bbox_info_idx['plate_color'])

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
                bbox_state_dict['start_frame'] = 0                                  # 车辆驶入画面的帧数
                bbox_state_dict['end_frame'] = 0                                    # 车辆驶出画面的帧数
                bbox_state_dict['attri_list'] = []                                  # 车辆属性列表
                bbox_state_dict['plate_color_list'] = []                            # 车牌颜色列表
                bbox_state_dict['car_disappear_frame_num'] = 0                      # 车辆消失画面帧数
                bbox_state_dict['up_report_flage'] = False                          # 上行抓拍标志位
                bbox_state_dict['down_report_flage'] = False                        # 下行抓拍标志位
                bbox_state_dict['left_report_flage'] = False                        # 左行抓拍标志位
                bbox_state_dict['right_report_flage'] = False                       # 右行抓拍标志位

                bbox_state_dict['id'] = bbox_info_idx['id']
                bbox_state_dict['loc'] = bbox_info_idx['loc']
                bbox_state_dict['loc_list'].append(bbox_info_idx['loc'])
                bbox_state_dict['stable_loc'] = bbox_info_idx['loc']
                bbox_state_dict['start_frame'] = frame_idx

                bbox_info_idx['stable_loc'] = bbox_info_idx['loc']

                # 更新车牌属性
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

        # 上下限阈值
        if self.roi_bool:
            Up_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_ratio[0]
            Down_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_ratio[1]
        else:
            Up_threshold = self.image_height * self.capture_line_ratio[0]
            Down_threshold = self.image_height * self.capture_line_ratio[1]

        # 报警逻辑
        # 1、快速抓怕：车辆行驶到触发线，执行抓拍。（近处、远处、左边、右边）
        for _, bbox_state_idy in self.params_dict['bbox_state_container'].items():
            # init 
            down_flage = False
            up_flage = False
            left_flage = False
            right_flage = False
            report_flage = False

            car_left_y = bbox_state_idy['loc'][0]
            car_right_y = bbox_state_idy['loc'][2]
            car_bottom_y = bbox_state_idy['loc'][3]

            # 如果车辆向下行驶, bbox_state_idy['up_down_state_frame_num'] >= 3 条件用于避免刚进 ROI 或者车辆静止状态下的误判
            if bbox_state_idy['up_down_state'] == 'Down' and bbox_state_idy['up_down_state_frame_num'] >= 3:
                if abs(car_bottom_y - Down_threshold) < self.capture_plate_up_down_distance_boundary_threshold:
                    down_flage = True

            # 如果车辆向上行驶，bbox_state_idy['up_down_state_frame_num'] 条件用于避免刚进 ROI 或者车辆静止状态下的误判
            if bbox_state_idy['up_down_state'] == 'Up' and bbox_state_idy['up_down_state_frame_num'] >= 3:
                if abs(car_bottom_y - Up_threshold) < self.capture_plate_up_down_distance_boundary_threshold:
                    up_flage = True

            # 如果车辆向左边行驶
            if bbox_state_idy['left_right_state'] == 'Left' and bbox_state_idy['left_right_state_frame_num'] >= 3:
                if abs(car_left_y - 0) < self.capture_plate_left_right_distance_boundary_threshold:
                    left_flage = True

            # 如果车辆向右边行驶
            if bbox_state_idy['left_right_state'] == 'Right' and bbox_state_idy['left_right_state_frame_num'] >= 3:
                if abs(self.image_width - car_right_y) < self.capture_plate_left_right_distance_boundary_threshold:
                    right_flage = True

            # 更新 capture_dict 抓拍字典
            capture_dict = {}                                                   # 抓怕
            capture_dict['id'] = bbox_state_idy['id']                           # 抓怕id
            capture_dict['flage'] = ''                                          # 抓拍标志信息
            capture_dict['capture_frame_num'] = 0                               # 抓拍帧数
            capture_dict['capture_bool'] = False                                # 抓拍成功标志

            if down_flage and not bbox_state_idy['down_report_flage']:
                bbox_state_idy['down_report_flage'] = True
                bbox_state_idy['left_report_flage'] = True
                bbox_state_idy['right_report_flage'] = True
                report_flage = True
                capture_dict['flage'] = 'down_flage'

            if up_flage and not bbox_state_idy['up_report_flage']:
                bbox_state_idy['up_report_flage'] = True
                bbox_state_idy['left_report_flage'] = True
                bbox_state_idy['right_report_flage'] = True
                report_flage = True
                capture_dict['flage'] = 'up_flage'

            if left_flage and not bbox_state_idy['left_report_flage']:
                bbox_state_idy['left_report_flage'] = True
                bbox_state_idy['down_report_flage'] = True
                bbox_state_idy['up_report_flage'] = True
                report_flage = True
                capture_dict['flage'] = 'left_flage'

            if right_flage and not bbox_state_idy['right_report_flage']:
                bbox_state_idy['right_report_flage'] = True
                bbox_state_idy['down_report_flage'] = True
                bbox_state_idy['up_report_flage'] = True
                report_flage = True
                capture_dict['flage'] = 'right_flage'

            if report_flage:
                # 更新 capture_dict 抓拍字典
                if bbox_state_idy['id'] not in self.params_dict['capture_dict']:

                    self.params_dict['capture_dict'][capture_dict['id']] = capture_dict                 # 抓拍序列
            
        return


    def update_capture_state(self, frame_idx):
        
        capture_res_list = []
        for capture_id_idx, capture_dict_idy in self.params_dict['capture_dict'].items():

            capture_flage_idx = capture_dict_idy['flage']

            # init 
            capture_res_dict = {}
            capture_res_dict['id'] = capture_id_idx
            capture_res_dict['start_frame'] = 0
            capture_res_dict['end_frame'] = frame_idx
            capture_res_dict['attri'] = 'None'
            capture_res_dict['plate_color'] = 'None'
            capture_res_dict['capture_flage'] = capture_flage_idx

            for _, bbox_state_idy in self.params_dict['bbox_state_container'].items():

                if bbox_state_idy['id'] == capture_id_idx:

                    # capture_flage
                    attri_np = np.array(bbox_state_idy['attri_list'])
                    plate_color_np = np.array(bbox_state_idy['plate_color_list'])
                    capture_dict_idy['capture_bool'] = True

                    if len(list(attri_np)):
                        capture_attri, capture_attri_frame = Counter(list(attri_np)).most_common(1)[0]
                        capture_res_dict['attri'] = capture_attri
                    if len(list(plate_color_np)):
                        capture_plate_color, capture_plate_color_frame = Counter(list(plate_color_np)).most_common(1)[0]
                        capture_res_dict['plate_color'] = capture_plate_color

                    capture_res_dict['start_frame'] = bbox_state_idy['start_frame']
                    capture_res_list.append(capture_res_dict)

        return capture_res_list