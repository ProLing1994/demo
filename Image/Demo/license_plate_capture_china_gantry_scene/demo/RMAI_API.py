from collections import Counter
import cv2
import numpy as np
import sys 
import random
# from torchstat import stat

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.detection2d.ssd_rfb_crossdatatraining.test_tools import SSDDetector
from Image.recognition2d.license_plate_ocr.infer.lpr import LPR
from Image.Demo.license_plate_capture_china_gantry_scene.sort.mot_sort import Sort


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
        
        # 5M
        # self.image_width = 2592
        # self.image_height = 1920

        # 2M
        self.image_width = 1920
        self.image_height = 1080

        # detector
        # 2022-03-09-17
        # pytorch 
        # self.ssd_car_plate_prototxt = None
        # self.ssd_car_plate_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-03-09-17_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
        # caffe
        # self.ssd_car_plate_prototxt = "/mnt/huanyuan/model_final/image_model/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-03-09-17/FPN_RFB_3class_3attri_noDilation_prior.prototxt"
        # self.ssd_car_plate_model_path = "/mnt/huanyuan/model_final/image_model/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-03-09-17/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-03-09-17.caffemodel"
        
        # 2022-04-25-18
        # pytorch 
        # self.ssd_car_plate_prototxt = None
        # self.ssd_car_plate_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-04-25-18_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
        # caffe
        # self.ssd_car_plate_prototxt = "/mnt/huanyuan/model_final/image_model/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-04-25-18/FPN_RFB_3class_3attri_noDilation_prior.prototxt"
        # self.ssd_car_plate_model_path = "/mnt/huanyuan/model_final/image_model/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-04-25-18/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-04-25-18.caffemodel"
        # openvino
        # self.ssd_car_plate_prototxt = None
        # self.ssd_car_plate_model_path = "/mnt/huanyuan/model_final/image_model/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-04-25-18/openvino_model/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-04-25-18.xml"

        # # 2022-05-27-00
        # # pytorch 
        # self.ssd_car_plate_prototxt = None
        # self.ssd_car_plate_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-05-27-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
        # # caffe
        # # self.ssd_car_plate_prototxt = "/mnt/huanyuan/model_final/image_model/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-05-27-00/FPN_RFB_3class_3attri_noDilation_prior.prototxt"
        # # self.ssd_car_plate_model_path = "/mnt/huanyuan/model_final/image_model/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-05-27-00/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-05-27-00.caffemodel"

        # # 2022-07-22-00
        # # pytorch 
        # self.ssd_car_plate_prototxt = None
        # self.ssd_car_plate_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-07-22-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
        # # # caffe
        # # self.ssd_car_plate_prototxt = "/mnt/huanyuan/model_final/image_model/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-07-22-00/FPN_RFB_3class_3attri_noDilation_prior.prototxt"
        # # self.ssd_car_plate_model_path = "/mnt/huanyuan/model_final/image_model/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-07-22-00/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-07-22-00.caffemodel"
        # # openvino
        # # self.ssd_car_plate_prototxt = None
        # # self.ssd_car_plate_model_path = "/mnt/huanyuan/model_final/image_model/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-07-22-00/openvino_model/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-07-22-00.xml"

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

        self.ssd_caffe_bool = False
        self.ssd_openvino_bool = False

        # 是否将 car\bus\truck 合并为一类输出
        self.merge_class_bool = True
        self.merge_class_name = 'car_bus_truck'
        self.car_attri_name_list = [ 'car', 'bus', 'truck' ]
        self.license_plate_name = 'license_plate'

        # sort
        self.max_age = 10
        self.min_hits = 3 
        self.iou_threshold = 0.3

        # lincense plate reader
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
        self.bbox_state_container_length = 10       # 车辆框连续丢失上报，从容器中清除该车辆信息

        # 更新车辆行驶状态
        self.update_state_container_length = 1      # 车辆框坐标容器大小，用于判断车辆状态
        self.update_state_num_threshold = 5         # 车辆行驶状态计数最大值，用于记录车辆处于同一行驶状态的帧数
        self.update_state_threshold = 1
        self.update_state_stable_loc_alpha = float(0.6)   # 平滑车辆框参数

        # 报警时间长短
        self.capture_frame_num_threshold = 16

        # 是否通过 roi 区域屏蔽部分检测结果
        self.roi_bool = False
        # self.roi_bool = True
        # 2M：16:9
        # args.roi_area = [270, 270, 1650, 1080]
        # # 5M：16:9
        self.roi_area = [0, 462, 2592, 1920]

        # 车牌长宽阈值
        # 2M：
        # self.plate_height = [20, 40]
        # self.plate_width = [50, 130]
        # 5M：
        self.plate_height = [20, 70]
        self.plate_width = [65, 170]

        # 抓拍线
        self.capture_line_ratio = [0.1, 0.5, 0.9]
        self.capture_stop_frame_num_threshold = 750     # 同一ID车辆如果存在于画面中帧差大于等于730（大于30秒），同时车辆状态为Stop，那么认为该车处于静止状态
        self.capture_plate_frame_threshold = 5
        self.capture_plate_disappear_frame_threshold = 5
        self.capture_plate_up_down_distance_boundary_threshold = 50
        self.capture_plate_left_right_distance_boundary_threshold = 50
        self.capture_plate_ocr_score_threshold = 0.8
        self.capture_plate_ocr_frame_threshold = 4


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
        bbox_info_dict['plate_ocr'] = ''                                    # 车牌识别结果（单帧）
        bbox_info_dict['plate_ocr_score'] = 0.0                             # 车牌识别结果得分（单帧）
        bbox_info_dict['plate_crop_bool'] = False                           # 车牌剪裁标志位（是否在图像roi内）
        
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
        bbox_state_dict['stop_frame_num'] = 0                               # 车辆停止状态画面帧数
        bbox_state_dict['up_down_speed'] = 0                                # 车辆速度（上下行）
        bbox_state_dict['left_right_speed'] = 0                             # 车辆速度（左右行）
        bbox_state_dict['center_point_list'] = []                           # 车辆中心点轨迹（多帧）
        bbox_state_dict['plate_ocr_list'] = []                              # 车牌识别结果（多帧）
        bbox_state_dict['plate_ocr_score_list'] = []                        # 车牌识别结果得分（多帧）
        bbox_state_dict['car_disappear_frame_num'] = 0                      # 车辆消失画面帧数
        bbox_state_dict['plate_frame_num'] = 0                              # 车牌出现画面帧数
        bbox_state_dict['plate_disappear_frame_num'] = 0                    # 车牌消失画面帧数
        bbox_state_dict['up_report_flage'] = False                          # 上行抓拍标志位
        bbox_state_dict['down_report_flage'] = False                        # 下行抓拍标志位
        bbox_state_dict['left_report_flage'] = False                        # 左行抓拍标志位
        bbox_state_dict['right_report_flage'] = False                       # 右行抓拍标志位
        bbox_state_dict['middle_report_flage'] = False                      # 过中线抓拍标志位
        bbox_state_dict['outtime_report_flage'] = False                     # 车辆超时标志位
        
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
            capture_line = [ self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * ratio for ratio in self.capture_line_ratio ]
        else:
            capture_line = [ self.image_height * ratio for ratio in self.capture_line_ratio ]

        self.update_capture_dict()
        capture_dict = self.params_dict['capture_dict']
        capture_res_list = self.update_capture_state( )

        return tracker_bboxes, bbox_info_list, bbox_state_map, capture_line, capture_dict, capture_res_list


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

            # # 计算车牌检测框与车辆检测框的交集区域，大于 0.2 则认为该车牌属于该车辆
            # if intersect_iou > 0.2:
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
        # gray img for lincense plate reader
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
            bbox_info_dict['plate_ocr'] = ''                                    # 车牌识别结果（单帧）
            bbox_info_dict['plate_ocr_score'] = 0.0                             # 车牌识别结果得分（单帧）
            bbox_info_dict['plate_crop_bool'] = False                           # 车牌剪裁标志位（是否在图像roi内）

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
                    Latent_plate = match_license_plate_roi[0][0:4]

                    # 按照车牌宽高过滤车牌
                    if (Latent_plate[3] - Latent_plate[1] > self.plate_height[0]) and \
                        (Latent_plate[3] - Latent_plate[1] < self.plate_height[1]) and \
                        (Latent_plate[2] - Latent_plate[0] > self.plate_width[0]) and \
                        (Latent_plate[2] - Latent_plate[0] < self.plate_width[1]):

                        bbox_info_dict['plate_loc'] = Latent_plate
                        
                        # lincense plate reader
                        # crop
                        crop_img = gray_img[bbox_info_dict['plate_loc'][1]:bbox_info_dict['plate_loc'][3], bbox_info_dict['plate_loc'][0]:bbox_info_dict['plate_loc'][2]]

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
            if bbox_state_idy['car_disappear_frame_num'] > self.bbox_state_container_length:
                pop_key_list.append(key)
            
            bbox_state_idy['car_disappear_frame_num'] += 1
        
        # pop
        for idx in range(len(pop_key_list)):
            # 更新 capture_dict 抓拍字典
            if pop_key_list[idx] not in self.params_dict['capture_dict']:
                bbox_state_idy = self.params_dict['bbox_state_container'][pop_key_list[idx]]
                if bbox_state_idy['middle_report_flage'] and \
                        not bbox_state_idy['down_report_flage'] and not bbox_state_idy['up_report_flage'] and \
                        not bbox_state_idy['left_report_flage'] and not bbox_state_idy['right_report_flage'] and \
                        not bbox_state_idy['outtime_report_flage']:
                    capture_dict = {}                                                   # 抓怕
                    capture_dict['id'] = bbox_state_idy['id']                           # 抓怕id
                    capture_dict['flage'] = 'middle_flage'                              # 抓拍标志信息
                    capture_dict['capture_frame_num'] = 0                               # 抓拍帧数
                    capture_dict['capture_bool'] = False                                # 抓拍成功标志
                    self.params_dict['capture_dict'][pop_key_list[idx]] = capture_dict  # 抓拍序列
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

                    # 车辆框漏检状态更新
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

                    # 车辆状态更新，判断静止时长
                    if bbox_state_idy['up_down_state'] == "Stop" and bbox_state_idy['left_right_state'] == "Stop":
                        bbox_state_idy['stop_frame_num'] += 1
                    else:
                        bbox_state_idy['stop_frame_num'] = 0
                    
                    car_bottom_y = bbox_state_idy['loc'][3]

                    # 更新车牌识别结果
                    if not bbox_info_idx['plate_ocr'] == '':
                        bbox_state_idy['plate_frame_num'] += 1
                        bbox_state_idy['plate_disappear_frame_num'] = 0

                        if self.roi_bool:
                            # 进入 roi 区域，在记录车牌信息
                            # 原因：远处车牌太小，结果不可信
                            if car_bottom_y > self.roi_area[1]:
                                bbox_state_idy['plate_ocr_list'].append(bbox_info_idx['plate_ocr'])
                                bbox_state_idy['plate_ocr_score_list'].append(bbox_info_idx['plate_ocr_score'])   
                                bbox_info_idx['plate_crop_bool'] = True
                        else:
                            bbox_state_idy['plate_ocr_list'].append(bbox_info_idx['plate_ocr'])
                            bbox_state_idy['plate_ocr_score_list'].append(bbox_info_idx['plate_ocr_score'])
                            bbox_info_idx['plate_crop_bool'] = True
                    else:
                        bbox_state_idy['plate_disappear_frame_num'] += 1

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
                bbox_state_dict['stop_frame_num'] = 0                               # 车辆停止状态画面帧数
                bbox_state_dict['up_down_speed'] = 0                                # 车辆速度（上下行）
                bbox_state_dict['left_right_speed'] = 0                             # 车辆速度（左右行）
                bbox_state_dict['center_point_list'] = []                           # 车辆中心点轨迹（多帧）
                bbox_state_dict['plate_ocr_list'] = []                              # 车牌识别结果（多帧）
                bbox_state_dict['plate_ocr_score_list'] = []                        # 车牌识别结果得分（多帧）
                bbox_state_dict['car_disappear_frame_num'] = 0                      # 车辆消失画面帧数
                bbox_state_dict['plate_frame_num'] = 0                              # 车牌出现画面帧数
                bbox_state_dict['plate_disappear_frame_num'] = 0                    # 车牌消失画面帧数
                bbox_state_dict['up_report_flage'] = False                          # 上行抓拍标志位
                bbox_state_dict['down_report_flage'] = False                        # 下行抓拍标志位
                bbox_state_dict['left_report_flage'] = False                        # 左行抓拍标志位
                bbox_state_dict['right_report_flage'] = False                       # 右行抓拍标志位
                bbox_state_dict['middle_report_flage'] = False                      # 过中线抓拍标志位
                bbox_state_dict['outtime_report_flage'] = False                     # 车辆超时标志位

                bbox_state_dict['id'] = bbox_info_idx['id']
                bbox_state_dict['loc'] = bbox_info_idx['loc']
                bbox_state_dict['loc_list'].append(bbox_info_idx['loc'])
                bbox_state_dict['stable_loc'] = bbox_info_idx['loc']

                bbox_info_idx['stable_loc'] = bbox_info_idx['loc']

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

        # 上下限阈值
        if self.roi_bool:
            Up_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_ratio[0]
            Middle_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_ratio[1]
            Down_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_ratio[2]
        else:
            Up_threshold = self.image_height * self.capture_line_ratio[0]
            Middle_threshold = self.image_height * self.capture_line_ratio[1]
            Down_threshold = self.image_height * self.capture_line_ratio[2]

        # 报警逻辑
        # 1、快速抓怕：车辆行驶到触发线，执行抓拍。（近处、远处、左边、右边）
        for _, bbox_state_idy in self.params_dict['bbox_state_container'].items():
            # init 
            down_flage = False
            up_flage = False
            left_flage = False
            right_flage = False
            middle_flage = False
            outtime_flage = False
            report_flage = False

            car_left_y = bbox_state_idy['loc'][0]
            car_right_y = bbox_state_idy['loc'][2]
            car_bottom_y = bbox_state_idy['loc'][3]

            # 如果车辆向下行驶, bbox_state_idy['up_down_state_frame_num'] >= 3 条件用于避免刚进 ROI 或者车辆静止状态下的误判
            if bbox_state_idy['up_down_state'] == 'Down' and bbox_state_idy['up_down_state_frame_num'] >= 3:
                if abs(car_bottom_y - Down_threshold) < self.capture_plate_up_down_distance_boundary_threshold and \
                    bbox_state_idy['plate_frame_num'] > self.capture_plate_frame_threshold:
                    down_flage = True
                else:
                    if car_bottom_y > Middle_threshold:
                        if bbox_state_idy['plate_disappear_frame_num'] == 0 and bbox_state_idy['plate_frame_num'] > self.capture_plate_frame_threshold:
                            middle_flage = True            # 此时车辆行驶过中线，处于需要报警状态

            # 如果车辆向上行驶，bbox_state_idy['up_down_state_frame_num'] 条件用于避免刚进 ROI 或者车辆静止状态下的误判
            if bbox_state_idy['up_down_state'] == 'Up' and bbox_state_idy['up_down_state_frame_num'] >= 3:
                if abs(car_bottom_y - Up_threshold) < self.capture_plate_up_down_distance_boundary_threshold and \
                    bbox_state_idy['plate_frame_num'] > self.capture_plate_frame_threshold:
                    up_flage = True
                else:
                    if car_bottom_y < Middle_threshold:
                        if bbox_state_idy['plate_disappear_frame_num'] == 0 and bbox_state_idy['plate_frame_num'] > self.capture_plate_frame_threshold:
                            middle_flage = True            # 此时车辆行驶过中线，处于需要报警状态

            # 如果车辆向左边行驶
            if bbox_state_idy['left_right_state'] == 'Left' and bbox_state_idy['left_right_state_frame_num'] >= 3:
                if abs(car_left_y - 0) < self.capture_plate_left_right_distance_boundary_threshold and \
                    bbox_state_idy['plate_frame_num'] > self.capture_plate_frame_threshold:
                    left_flage = True

            # 如果车辆向右边行驶
            if bbox_state_idy['left_right_state'] == 'Right' and bbox_state_idy['left_right_state_frame_num'] >= 3:
                if abs(self.image_width - car_right_y) < self.capture_plate_left_right_distance_boundary_threshold and \
                    bbox_state_idy['plate_frame_num'] > self.capture_plate_frame_threshold:
                    right_flage = True

            if ( bbox_state_idy['up_down_state'] == 'Stop' and bbox_state_idy['up_down_state_frame_num'] >= 3 ) and \
                ( bbox_state_idy['left_right_state'] == 'Stop' and bbox_state_idy['left_right_state_frame_num'] >= 3 ):
                if bbox_state_idy['stop_frame_num'] > self.capture_stop_frame_num_threshold:
                    if bbox_state_idy['plate_disappear_frame_num'] == 0 and bbox_state_idy['plate_frame_num'] > self.capture_plate_frame_threshold:
                        outtime_flage = True

            # 更新 capture_dict 抓拍字典
            capture_dict = {}                                                   # 抓怕
            capture_dict['id'] = bbox_state_idy['id']                           # 抓怕id
            capture_dict['flage'] = ''                                          # 抓拍标志信息
            capture_dict['capture_frame_num'] = 0                               # 抓拍帧数
            capture_dict['capture_bool'] = False                                # 抓拍成功标志

            if down_flage and not bbox_state_idy['down_report_flage']:
                bbox_state_idy['down_report_flage'] = True
                bbox_state_idy['middle_report_flage'] = False
                bbox_state_idy['left_report_flage'] = True
                bbox_state_idy['right_report_flage'] = True
                report_flage = True
                capture_dict['flage'] = 'down_flage'

            if up_flage and not bbox_state_idy['up_report_flage']:
                bbox_state_idy['up_report_flage'] = True
                bbox_state_idy['middle_report_flage'] = False
                bbox_state_idy['left_report_flage'] = True
                bbox_state_idy['right_report_flage'] = True
                report_flage = True
                capture_dict['flage'] = 'up_flage'

            if left_flage and not bbox_state_idy['left_report_flage']:
                bbox_state_idy['left_report_flage'] = True
                bbox_state_idy['middle_report_flage'] = False
                bbox_state_idy['down_report_flage'] = True
                bbox_state_idy['up_report_flage'] = True
                report_flage = True
                capture_dict['flage'] = 'left_flage'

            if right_flage and not bbox_state_idy['right_report_flage']:
                bbox_state_idy['right_report_flage'] = True
                bbox_state_idy['middle_report_flage'] = False
                bbox_state_idy['down_report_flage'] = True
                bbox_state_idy['up_report_flage'] = True
                report_flage = True
                capture_dict['flage'] = 'right_flage'

            if outtime_flage and not bbox_state_idy['outtime_report_flage']: 
                bbox_state_idy['outtime_report_flage'] = True
                bbox_state_idy['middle_report_flage'] = False
                report_flage = True
                capture_dict['flage'] = 'outtime_flage'

            if middle_flage and not bbox_state_idy['middle_report_flage']:
                bbox_state_idy['middle_report_flage'] = True

            if report_flage:
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
                                if capture_flage_idx == 'down_flage':
                                    capture_res_dict['plate_state'] = 'down'

                                # 抓到车牌，标志位置1
                                if capture_flage_idx == 'up_flage':
                                    capture_res_dict['plate_state'] = 'up'

                                # 抓到车牌，标志位置1
                                if capture_flage_idx == 'left_flage':
                                    capture_res_dict['plate_state'] = 'left'

                                # 抓到车牌，标志位置1
                                if capture_flage_idx == 'right_flage':
                                    capture_res_dict['plate_state'] = 'right'

                                # 抓到车牌，标志位置1
                                if capture_flage_idx == 'outtime_flage':
                                    capture_res_dict['plate_state'] = 'outtime'

                                # 抓到车牌，标志位置1
                                if capture_flage_idx == 'middle_flage':
                                    capture_res_dict['plate_state'] = 'middle'
                                    
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