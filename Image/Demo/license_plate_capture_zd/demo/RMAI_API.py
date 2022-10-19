from collections import Counter
import cv2
import copy
import numpy as np
import sys 
import random

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Demo.license_plate_capture_zd.model.LPR_detect import LPRDetectCaffe, LPRDetectOpenVINO
from Image.detection2d.mmdetection.demo.detector.yolov6_detector import YOLOV6Detector
from Image.recognition2d.lpr.infer.lpr_seg_ocr import LPRSegOcrcffe
from Image.Demo.license_plate_capture_zd.sort.mot_sort import Sort


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

        # detector
        # ssd
        self.ssd_bool = True
        # # pytorch 
        # self.ssd_plate_prototxt = None
        # self.ssd_plate_model_path = ""
        # # caffe
        self.ssd_plate_prototxt = "/mnt/huanyuan/model_final/image_model/zd_ssd_rfb_wmr/ssd_mbv2_2class/caffe_model/ssd_mobilenetv2_fpn.prototxt"
        self.ssd_plate_model_path = "/mnt/huanyuan/model_final/image_model/zd_ssd_rfb_wmr/ssd_mbv2_2class/caffe_model/ssd_mobilenetv2_0421.caffemodel"
        # openvino
        # self.ssd_plate_model_path = "/mnt/huanyuan/model_final/image_model/zd_ssd_rfb_wmr/ssd_mbv2_2class/openvino_model/ssd_mobilenetv2_fpn.xml"
        self.ssd_caffe_bool = True
        self.ssd_openvino_bool = False

        # yolov6
        self.yolov6_bool = False
        # pytorch
        self.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_zd_plate_wmr/yolov6_licenseplate_deploy.py"
        self.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_zd_plate_wmr/epoch_95_deploy.pth"
        
        self.detect_class_name = ['license_plate']
        self.detect_class_threshold_list = [0.4]

        # lpr
        # seg: zd
        self.lpr_seg_zd_caffe_prototxt = "/mnt/huanyuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_0826/LaneNetNova_class_15.prototxt"
        self.lpr_seg_zd_caffe_model_path = "/mnt/huanyuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_0826/LaneNetNova_seg_city_cartype_kind_num_zd_0826.caffemodel"
        # # ocr: zd 0901
        # self.lpr_ocr_zd_caffe_prototxt = "/mnt/huanyuan/model/image/lpr/zd/ocr_zd_mask_all_UAE_0901/cnn_256x64_38.prototxt"
        # self.lpr_ocr_zd_caffe_model_path = "/mnt/huanyuan/model/image/lpr/zd/ocr_zd_mask_all_UAE_0901/ocr_zd_mask_UAE_0901.caffemodel"
        # ocr: zd 0901
        self.lpr_ocr_zd_caffe_prototxt = "/mnt/huanyuan/model/image/lpr/zd/ocr_zd_mask_all_UAE_hisi_1010/cnn_256x64_38.prototxt"
        self.lpr_ocr_zd_caffe_model_path = "/mnt/huanyuan/model/image/lpr/zd/ocr_zd_mask_all_UAE_hisi_1010/ocr_zd_mask_UAE_hisi_1010.caffemodel"

        # lpr params
        self.lpr_ocr_width_expand_ratio = 0.05
        self.lpr_ocr_column_threshold = 2.5

        # sort
        self.max_age = 10
        self.min_hits = 3 
        self.iou_threshold = 0.1
        self.sort_expand_ratio = 1.5

        # 缓存间隔
        self.cache_interval = 2
        # 缓存容器长度
        self.cache_container_length = 8

        # 状态容器长度
        self.bbox_state_container_length = 10       # 车牌框连续丢失上报，从容器中清除该车辆信息
        self.lpr_ocr_state_container_length = 20    # 车牌状态长度阈值

        # 更新车辆行驶状态
        self.update_state_num_threshold = 5         # 行驶状态计数最大值，用于记录车辆处于同一行驶状态的帧数
        self.update_state_threshold = 1
        self.update_state_stable_loc_alpha = float(0.6)   # 平滑检测框参数

        # 报警时间长短
        self.capture_frame_num_threshold = 3
        self.capture_clear_frame_num_threshold = 60 * 25        # 经过多少帧，抓拍容器清空

        # 是否通过 roi 区域屏蔽部分检测结果
        self.roi_bool = False
        # self.roi_bool = True
        self.roi_area = [0, 0, self.image_width, self.image_height]

        # 车牌长宽阈值
        self.plate_signel_height = [25, 960]
        self.plate_signel_width = [0, 1920]
        self.plate_double_height = [45, 960]
        self.plate_double_width = [0, 1920]

        # 抓拍线
        self.capture_line_up_down_ratio = [0.02, 0.5, 0.9, 0.98]
        self.capture_line_left_right_ratio = [0.01, 0.25, 0.75, 0.99]
        self.capture_plate_frame_threshold = 5
        self.capture_outtime_frame_threshold_01 = 25
        self.capture_plate_up_down_distance_boundary_threshold = 100
        self.capture_plate_left_right_distance_near_boundary_threshold = 100
        self.capture_plate_left_right_distance_far_boundary_threshold = 400
        self.capture_plate_ocr_score_threshold = 0.8
        self.capture_lpr_contry_frame_threshold = 3
        self.capture_lpr_city_frame_threshold = 2
        self.capture_lpr_car_type_frame_threshold = 3
        self.capture_lpr_kind_frame_threshold = 4
        self.capture_lpr_num_frame_threshold = 4

        # 上下限阈值 & 左右限阈值
        if self.roi_bool:
            self.ROI_Up_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_up_down_ratio[0]
            self.ROI_Down_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_up_down_ratio[3]
            self.Up_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_up_down_ratio[1]
            self.Down_threshold = self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * self.capture_line_up_down_ratio[2]
            self.ROI_Left_threshold = self.roi_area[0] + ( self.roi_area[2] - self.roi_area[0] ) * self.capture_line_left_right_ratio[0]
            self.ROI_Right_threshold = self.roi_area[0] + ( self.roi_area[2] - self.roi_area[0] ) * self.capture_line_left_right_ratio[3]
            self.Left_threshold = self.roi_area[0] + ( self.roi_area[2] - self.roi_area[0] ) * self.capture_line_left_right_ratio[1]
            self.Right_threshold = self.roi_area[0] + ( self.roi_area[2] - self.roi_area[0] ) * self.capture_line_left_right_ratio[2]
        else:
            self.ROI_Up_threshold = self.image_height * self.capture_line_up_down_ratio[0]
            self.ROI_Down_threshold = self.image_height * self.capture_line_up_down_ratio[3]
            self.Up_threshold = self.image_height * self.capture_line_up_down_ratio[1]
            self.Down_threshold = self.image_height * self.capture_line_up_down_ratio[2]
            self.ROI_Left_threshold = self.image_width * self.capture_line_left_right_ratio[0]
            self.ROI_Right_threshold = self.image_width * self.capture_line_left_right_ratio[3]
            self.Left_threshold = self.image_width * self.capture_line_left_right_ratio[1]
            self.Right_threshold = self.image_width * self.capture_line_left_right_ratio[2]


    def param_init(self):
        self.params_dict = {}

        # bbox_info_dict
        bbox_info_dict = {}
        bbox_info_dict['id'] = 0                                            # 追踪id
        bbox_info_dict['loc'] = []                                          # 车牌坐标
        bbox_info_dict['kind_loc'] = [0,0,0,0]                              # 车牌编号坐标
        bbox_info_dict['num_loc'] = [0,0,0,0]                               # 车牌号码坐标
        bbox_info_dict['country'] = 'None'                                  # 车牌国家
        bbox_info_dict['city'] = 'None'                                     # 车牌城市
        bbox_info_dict['car_type'] = 'None'                                 # 车牌车型
        bbox_info_dict['kind'] = 'None'                                     # 车牌编号
        bbox_info_dict['num'] = 'None'                                      # 车牌号码
        bbox_info_dict['column'] = 'None'                                   # 车牌单双行
        bbox_info_dict['score'] = 0.0                                       # 车牌得分
        bbox_info_dict['ignore'] = False                                    # 车牌忽略（太小）
        bbox_info_dict['frame_num'] = 0                                     # 车牌进入画面帧数
        bbox_info_dict['up_down_speed'] = 0                                 # 车牌速度（上下行）
        bbox_info_dict['left_right_speed'] = 0                              # 车牌速度（左右行
        bbox_info_dict['up_down_state'] = 'Stop'                            # 车牌状态（上下行）
        bbox_info_dict['up_down_state_frame_num'] = 0                       # 车牌状态（上下行）帧数
        bbox_info_dict['left_right_state'] = 'Stop'                         # 车牌状态（左右行）
        bbox_info_dict['left_right_state_frame_num'] = 0                    # 车牌状态（左右行）帧数
        bbox_info_dict['lpr_num'] = 0                                       # 车牌识别帧数

        self.params_dict['bbox_info_container'] = []                        # 缓存容器（imag & bbox_info_dict）

        # bbox_state_dict
        bbox_state_dict = {}
        bbox_state_dict['id'] = 0                                           # 追踪id
        bbox_state_dict['loc'] = []                                         # 车牌坐标
        bbox_state_dict['stable_loc'] = []                                  # 车牌坐标（稳定）
        bbox_state_dict['center_point_list'] = []                           # 车牌中心点轨迹（多帧）
        bbox_state_dict['frame_num'] = 0                                    # 车牌进入画面帧数
        bbox_state_dict['disappear_frame_num'] = 0                          # 车牌消失画面帧数
        bbox_state_dict['up_down_speed'] = 0                                # 车牌速度（上下行）
        bbox_state_dict['left_right_speed'] = 0                             # 车牌速度（左右行）
        bbox_state_dict['up_down_state'] = 'Stop'                           # 车牌状态（上下行）
        bbox_state_dict['up_down_state_frame_num'] = 0                      # 车牌状态（上下行）帧数
        bbox_state_dict['left_right_state'] = 'Stop'                        # 车牌状态（左右行）
        bbox_state_dict['left_right_state_frame_num'] = 0                   # 车牌状态（左右行）帧数
        bbox_state_dict['lpr_num'] = 0                                      # 车牌识别帧数
        bbox_state_dict['lpr_country_list'] = []                            # 车牌国家结果（多帧）
        bbox_state_dict['lpr_city_list'] = []                               # 车牌城市结果（多帧）
        bbox_state_dict['lpr_car_type_list'] = []                           # 车牌车型结果（多帧）
        bbox_state_dict['lpr_kind_list'] = []                               # 车牌编号结果（多帧）
        bbox_state_dict['lpr_num_list'] = []                                # 车牌号码结果（多帧）
        bbox_state_dict['lpr_column_list'] = []                             # 车牌单双行结果（多帧）
        bbox_state_dict['lpr_score_list'] = []                              # 车牌识别得分（多帧）
        bbox_state_dict['far_report_flage'] = False                         # 抓拍标志位
        bbox_state_dict['near_report_flage'] = False                        # 抓拍标志位
        bbox_state_dict['left_report_flage'] = False                        # 抓拍标志位
        bbox_state_dict['right_report_flage'] = False                       # 抓拍标志位
        bbox_state_dict['outtime_flage_01'] = False                         # 抓拍标志位
        
        self.params_dict['bbox_state_container'] = {}                       # 状态信息容器（key: 追踪id, value: bbox_state_dict）

        # capture_dict
        capture_dict = {}                                                   
        capture_dict['id'] = 0                                              # 抓拍id
        capture_dict['flage'] = ''                                          # 抓拍标志信息
        capture_dict['capture_frame_num'] = 0                               # 抓拍帧数
        capture_dict['capture_bool'] = False                                # 抓拍成功标志

        self.params_dict['capture_container'] = {}                          # 抓拍序列容器（key: 追踪id, value: capture_dict）

        # capture_res_dict
        capture_res_dict = {}
        capture_res_dict['id'] = 0                                          # 抓拍id
        capture_res_dict['country'] = 'None'                                # 车牌国家结果
        capture_res_dict['city'] = 'None'                                   # 车牌城市结果
        capture_res_dict['car_type'] = 'None'                               # 车牌车型结果
        capture_res_dict['kind'] = 'None'                                   # 车牌编号结果
        capture_res_dict['num'] = 'None'                                    # 车牌号码结果
        capture_res_dict['column'] = 'None'                                 # 车牌单双行
        capture_res_dict['flage'] = ''                                      # 抓拍标志信息
        capture_res_dict['capture_frame_num'] = 0                           # 抓拍帧数
        capture_res_dict['img_bbox_info_list'] = []                         # 抓拍结果
        capture_res_dict['draw_bool'] = False                               # 绘图标志

        self.params_dict['capture_res_container'] = {}                      # 抓拍结果容器（key: 车牌号码结果, value: capture_res_dict）


    def model_init(self):
        
        # detector
        if self.ssd_bool:
            if self.ssd_caffe_bool:
                self.detector = LPRDetectCaffe(self.ssd_plate_prototxt, self.ssd_plate_model_path)
            elif self.ssd_openvino_bool:
                self.detector = LPRDetectOpenVINO(self.ssd_plate_model_path)

        elif self.yolov6_bool:
            self.detector = YOLOV6Detector(self.yolov6_config, self.yolov6_checkpoint, class_name=self.detect_class_name, threshold_list=self.detect_class_threshold_list)

        # tracker
        self.mot_tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_threshold)

        # lincense plate seg
        # self.lpr_seg = LPRSegCaffe(self.lpr_seg_zd_caffe_prototxt, self.lpr_seg_zd_caffe_model_path)
        self.lpr_seg_ocr = LPRSegOcrcffe(self.lpr_seg_zd_caffe_prototxt, self.lpr_seg_zd_caffe_model_path, self.lpr_ocr_zd_caffe_prototxt, self.lpr_ocr_zd_caffe_model_path)


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
        tracker_bboxes = self.update_tracker_bboxes( copy.deepcopy(bboxes) )

        # update bbox info
        bbox_info_list = self.update_bbox_info( tracker_bboxes )

        # update plate info
        bbox_info_list = self.update_plate_info( img, bbox_info_list )

        # store
        # 跳帧存储原图和检测识别结果
        self.update_cache_container( img, frame_idx, bbox_info_list )

        # 更新状态容器，同时更新车辆行驶状态和帧率
        bbox_info_list = self.update_bbox_state_container( bbox_info_list )
        bbox_state_map = self.params_dict['bbox_state_container']

        # captute
        self.update_capture_dict()
        capture_list = self.params_dict['capture_container']
        self.update_capture_state()
        capture_res_list = self.params_dict['capture_res_container']

        ## capture_line
        if self.roi_bool:
            capture_line_up_down = [ self.roi_area[1] + ( self.roi_area[3] - self.roi_area[1] ) * ratio for ratio in self.capture_line_up_down_ratio ]
            capture_line_left_right = [ self.roi_area[0] + ( self.roi_area[2] - self.roi_area[0] ) * ratio for ratio in self.capture_line_left_right_ratio ]
        else:
            capture_line_up_down = [ self.image_height * ratio for ratio in self.capture_line_up_down_ratio ]
            capture_line_left_right = [ self.image_width * ratio for ratio in self.capture_line_left_right_ratio ]

        return bboxes, tracker_bboxes, bbox_info_list, bbox_state_map, capture_line_up_down, capture_line_left_right, capture_list, capture_res_list


    def update_tracker_bboxes(self, bboxes):
    
        # tracker
        dets = np.empty((0, 5))
        for idx in range(len(self.detect_class_name)):
            detect_class_name_idx = self.detect_class_name[idx]
            if detect_class_name_idx in bboxes:

                # license_plate
                # 由于车牌面积太小，跟踪不上，需要增大面积
                bboxes_class_list = bboxes[detect_class_name_idx]
                for idy in range(len(bboxes_class_list)):
                    bboxes_idx = bboxes_class_list[idy]
                    
                    bboxes_idx_width = ( bboxes_idx[2] - bboxes_idx[0] ) * self.sort_expand_ratio
                    bboxes_idx_height = ( bboxes_idx[3] - bboxes_idx[1] ) * self.sort_expand_ratio
                    bboxes_idx[0] = bboxes_idx[0] - bboxes_idx_width
                    bboxes_idx[2] = bboxes_idx[2] + bboxes_idx_width
                    bboxes_idx[1] = bboxes_idx[1] - bboxes_idx_height
                    bboxes_idx[3] = bboxes_idx[3] + bboxes_idx_height

                dets = np.concatenate((dets, np.array(bboxes_class_list)), axis=0)
        
        tracker_bboxes = self.mot_tracker.update(dets)

        # license_plate
        # 由于增加了面积，需要减少面积，获得精确坐标
        for idx in range(len(tracker_bboxes)):
            bboxes_idx = tracker_bboxes[idx]

            bboxes_idx_width = (( bboxes_idx[2] - bboxes_idx[0] ) * self.sort_expand_ratio ) / ( 1.0 + self.sort_expand_ratio * 2.0 )
            bboxes_idx_height = (( bboxes_idx[3] - bboxes_idx[1] ) * self.sort_expand_ratio )  / ( 1.0 + self.sort_expand_ratio * 2.0 )
            bboxes_idx[0] = bboxes_idx[0] + bboxes_idx_width
            bboxes_idx[2] = bboxes_idx[2] - bboxes_idx_width
            bboxes_idx[1] = bboxes_idx[1] + bboxes_idx_height
            bboxes_idx[3] = bboxes_idx[3] - bboxes_idx_height

        return tracker_bboxes


    def update_bbox_info(self, tracker_bboxes):
        
        bbox_info_list = []
        for idx in range(len(tracker_bboxes)):
            # init 
            # bbox_info_dict
            bbox_info_dict = {}
            bbox_info_dict['id'] = 0                                            # 追踪id
            bbox_info_dict['loc'] = []                                          # 车牌坐标
            bbox_info_dict['kind_loc'] = [0,0,0,0]                              # 车牌编号坐标
            bbox_info_dict['num_loc'] = [0,0,0,0]                               # 车牌号码坐标
            bbox_info_dict['country'] = 'None'                                  # 车牌国家
            bbox_info_dict['city'] = 'None'                                     # 车牌城市
            bbox_info_dict['car_type'] = 'None'                                 # 车牌车型
            bbox_info_dict['kind'] = 'None'                                     # 车牌编号
            bbox_info_dict['num'] = 'None'                                      # 车牌号码
            bbox_info_dict['column'] = 'None'                                   # 车牌单双行
            bbox_info_dict['score'] = 0.0                                       # 车牌得分
            bbox_info_dict['ignore'] = False                                    # 车牌忽略（太小）
            bbox_info_dict['frame_num'] = 0                                     # 车牌进入画面帧数
            bbox_info_dict['up_down_speed'] = 0                                 # 车牌速度（上下行）
            bbox_info_dict['left_right_speed'] = 0                              # 车牌速度（左右行
            bbox_info_dict['up_down_state'] = 'Stop'                            # 车牌状态（上下行）
            bbox_info_dict['up_down_state_frame_num'] = 0                       # 车牌状态（上下行）帧数
            bbox_info_dict['left_right_state'] = 'Stop'                         # 车牌状态（左右行）
            bbox_info_dict['left_right_state_frame_num'] = 0                    # 车牌状态（左右行）帧数
            bbox_info_dict['lpr_num'] = 0                                       # 车牌识别帧数

            # license_plate
            tracker_bbox = tracker_bboxes[idx]
            bbox_info_dict['id'] = tracker_bbox[-1]
            bbox_info_dict['loc'] = tracker_bbox[0:4]

            bbox_info_list.append(bbox_info_dict)

        return bbox_info_list
    
    
    def update_plate_info( self, img, bbox_info_list ):

        # 遍历单帧结果
        for idx in range(len(bbox_info_list)):
            bbox_info_idx = bbox_info_list[idx]

            # lincense plate reader
            # crop
            x1 = min(max(0, int(bbox_info_idx['loc'][0] - self.lpr_ocr_width_expand_ratio * (bbox_info_idx['loc'][2] - bbox_info_idx['loc'][0]))), self.image_width)
            x2 = min(max(0, int(bbox_info_idx['loc'][2] + self.lpr_ocr_width_expand_ratio * (bbox_info_idx['loc'][2] - bbox_info_idx['loc'][0]))), self.image_width)
            y1 = min(max(0, int(bbox_info_idx['loc'][1])), self.image_height)
            y2 = min(max(0, int(bbox_info_idx['loc'][3])), self.image_height)
            crop_img = img[y1: y2, x1: x2]
            crop_img_aspect = crop_img.shape[1] / crop_img.shape[0]

            seg_mask, seg_bbox, seg_info, ocr, ocr_score, ocr_ignore = self.lpr_seg_ocr.run(crop_img)

            if 'kind' in seg_bbox:
                bbox_info_idx['kind_loc'][0] = x1 + seg_bbox['kind'][0][0]
                bbox_info_idx['kind_loc'][1] = y1 + seg_bbox['kind'][0][1]
                bbox_info_idx['kind_loc'][2] = x1 + seg_bbox['kind'][0][0] + seg_bbox['kind'][0][2]
                bbox_info_idx['kind_loc'][3] = y1 + seg_bbox['kind'][0][1] + seg_bbox['kind'][0][3]

            if 'num' in seg_bbox:
                bbox_info_idx['num_loc'][0] = x1 + seg_bbox['num'][0][0]
                bbox_info_idx['num_loc'][1] = y1 + seg_bbox['num'][0][1]
                bbox_info_idx['num_loc'][2] = x1 + seg_bbox['num'][0][0] + seg_bbox['num'][0][2]
                bbox_info_idx['num_loc'][3] = y1 + seg_bbox['num'][0][1] + seg_bbox['num'][0][3]

            bbox_info_idx['country'] = seg_info['country']
            bbox_info_idx['city'] = seg_info['city']
            bbox_info_idx['car_type'] = seg_info['car_type']
            bbox_info_idx['kind'] = seg_info['kind']
            bbox_info_idx['num'] = seg_info['num']
            bbox_info_idx['column'] = 'Single' if crop_img_aspect > self.lpr_ocr_column_threshold else 'Double'
            bbox_info_idx['score'] = np.array(ocr_score).mean() if len(ocr_score) else 0.0
            bbox_info_idx['ignore'] = ocr_ignore

            if seg_info['kind'] == 'kind':
                kind = ocr.split('#')[0] 
                if kind != "#":
                    bbox_info_idx['kind'] = kind
                
            if seg_info['num'] == 'num':  
                num = ocr.split('#')[-1] 
                if num != "#":
                    bbox_info_idx['num'] = num
        
        return bbox_info_list


    def update_cache_container(self, img, frame_idx, bbox_info_list):

        if frame_idx % self.cache_interval == 0:
            self.params_dict['bbox_info_container'].append({'img': img, 'bbox_info': bbox_info_list})

        if len(self.params_dict['bbox_info_container']) > self.cache_container_length:
            self.params_dict['bbox_info_container'].pop(0)
    

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
                    lpr_width =  bbox_state_idy['loc'][2] - bbox_state_idy['loc'][0]
                    lpr_height =  bbox_state_idy['loc'][3] - bbox_state_idy['loc'][1]

                    bool_add_lpr = False
                    if bbox_state_idy['loc'][0] > self.ROI_Left_threshold and bbox_state_idy['loc'][2] < self.ROI_Right_threshold and \
                        bbox_state_idy['loc'][1] > self.ROI_Up_threshold and bbox_state_idy['loc'][3] < self.ROI_Down_threshold and \
                        bbox_info_idx['ignore'] == False:

                            if bbox_info_idx['column'] == 'Single' and \
                                lpr_width >= self.plate_signel_width[0] and \
                                lpr_width < self.plate_signel_width[1] and \
                                lpr_height >= self.plate_signel_height[0] and \
                                lpr_height < self.plate_signel_height[1]:
                                bool_add_lpr = True

                            if bbox_info_idx['column'] == 'Double' and \
                                lpr_width >= self.plate_double_width[0] and \
                                lpr_width < self.plate_double_width[1] and \
                                lpr_height >= self.plate_double_height[0] and \
                                lpr_height < self.plate_double_height[1]:
                                bool_add_lpr = True

                    if bool_add_lpr:
                        bbox_state_idy['lpr_num'] += 1
                        bbox_state_idy['lpr_country_list'].append(bbox_info_idx['country'])
                        bbox_state_idy['lpr_city_list'].append(bbox_info_idx['city'])
                        bbox_state_idy['lpr_car_type_list'].append(bbox_info_idx['car_type'])
                        bbox_state_idy['lpr_kind_list'].append(bbox_info_idx['kind'])
                        bbox_state_idy['lpr_num_list'].append(bbox_info_idx['num'])
                        bbox_state_idy['lpr_column_list'].append(bbox_info_idx['column'])
                        bbox_state_idy['lpr_score_list'].append(bbox_info_idx['score'])

                        if len( bbox_state_idy['lpr_country_list'] ) > self.lpr_ocr_state_container_length:
                            bbox_state_idy['lpr_country_list'].pop(0)
                        if len( bbox_state_idy['lpr_city_list'] ) > self.lpr_ocr_state_container_length:
                            bbox_state_idy['lpr_city_list'].pop(0)
                        if len( bbox_state_idy['lpr_car_type_list'] ) > self.lpr_ocr_state_container_length:
                            bbox_state_idy['lpr_car_type_list'].pop(0)
                        if len( bbox_state_idy['lpr_kind_list'] ) > self.lpr_ocr_state_container_length:
                            bbox_state_idy['lpr_kind_list'].pop(0)
                        if len( bbox_state_idy['lpr_num_list'] ) > self.lpr_ocr_state_container_length:
                            bbox_state_idy['lpr_num_list'].pop(0)
                        if len( bbox_state_idy['lpr_column_list'] ) > self.lpr_ocr_state_container_length:
                            bbox_state_idy['lpr_column_list'].pop(0)
                        if len( bbox_state_idy['lpr_score_list'] ) > self.lpr_ocr_state_container_length:
                            bbox_state_idy['lpr_score_list'].pop(0)

                    # 信息同步
                    bbox_info_idx['frame_num'] = bbox_state_idy['frame_num']
                    bbox_info_idx['up_down_speed'] = bbox_state_idy['up_down_speed']
                    bbox_info_idx['left_right_speed'] = bbox_state_idy['left_right_speed']
                    bbox_info_idx['up_down_state'] = bbox_state_idy['up_down_state']
                    bbox_info_idx['up_down_state_frame_num'] = bbox_state_idy['up_down_state_frame_num']
                    bbox_info_idx['left_right_state'] = bbox_state_idy['left_right_state']
                    bbox_info_idx['left_right_state_frame_num'] = bbox_state_idy['left_right_state_frame_num']
                    bbox_info_idx['lpr_num'] = bbox_state_idy['lpr_num']

            if is_new_id_bool:

                # bbox_state_dict
                bbox_state_dict = {}
                bbox_state_dict['id'] = 0                                           # 追踪id
                bbox_state_dict['loc'] = []                                         # 车牌坐标
                bbox_state_dict['stable_loc'] = []                                  # 车牌坐标（稳定）
                bbox_state_dict['center_point_list'] = []                           # 车牌中心点轨迹（多帧）
                bbox_state_dict['frame_num'] = 0                                    # 车牌进入画面帧数
                bbox_state_dict['disappear_frame_num'] = 0                          # 车牌消失画面帧数
                bbox_state_dict['up_down_speed'] = 0                                # 车牌速度（上下行）
                bbox_state_dict['left_right_speed'] = 0                             # 车牌速度（左右行）
                bbox_state_dict['up_down_state'] = 'Stop'                           # 车牌状态（上下行）
                bbox_state_dict['up_down_state_frame_num'] = 0                      # 车牌状态（上下行）帧数
                bbox_state_dict['left_right_state'] = 'Stop'                        # 车牌状态（左右行）
                bbox_state_dict['left_right_state_frame_num'] = 0                   # 车牌状态（左右行）帧数
                bbox_state_dict['lpr_num'] = 0                                      # 车牌识别帧数
                bbox_state_dict['lpr_country_list'] = []                            # 车牌国家结果（多帧）
                bbox_state_dict['lpr_city_list'] = []                               # 车牌城市结果（多帧）
                bbox_state_dict['lpr_car_type_list'] = []                           # 车牌车型结果（多帧）
                bbox_state_dict['lpr_kind_list'] = []                               # 车牌编号结果（多帧）
                bbox_state_dict['lpr_num_list'] = []                                # 车牌号码结果（多帧）
                bbox_state_dict['lpr_column_list'] = []                             # 车牌单双行结果（多帧）
                bbox_state_dict['lpr_score_list'] = []                              # 车牌识别得分（多帧）
                bbox_state_dict['far_report_flage'] = False                         # 抓拍标志位
                bbox_state_dict['near_report_flage'] = False                        # 抓拍标志位
                bbox_state_dict['left_report_flage'] = False                        # 抓拍标志位
                bbox_state_dict['right_report_flage'] = False                       # 抓拍标志位
                bbox_state_dict['outtime_flage_01'] = False                         # 抓拍标志位

                bbox_state_dict['id'] = bbox_info_idx['id']
                bbox_state_dict['loc'] = bbox_info_idx['loc']
                bbox_state_dict['stable_loc'] = bbox_info_idx['loc']
                bbox_state_dict['frame_num'] += 1

                # 更新车牌识别有效帧数
                loc_center_x = (bbox_info_idx['loc'][0] + bbox_info_idx['loc'][2]) / 2.0
                loc_center_y = (bbox_info_idx['loc'][1] + bbox_info_idx['loc'][3]) / 2.0
                lpr_width =  bbox_info_idx['loc'][2] - bbox_info_idx['loc'][0]
                lpr_height =  bbox_info_idx['loc'][3] - bbox_info_idx['loc'][1]

                bool_add_lpr = False
                if bbox_state_idy['loc'][0] > self.ROI_Left_threshold and bbox_info_idx['loc'][2] < self.ROI_Right_threshold and \
                    bbox_info_idx['loc'][1] > self.ROI_Up_threshold and bbox_info_idx['loc'][3] < self.ROI_Down_threshold and \
                    bbox_info_idx['ignore'] == False:

                        if bbox_info_idx['column'] == 'Single' and \
                            lpr_width >= self.plate_signel_width[0] and \
                            lpr_width < self.plate_signel_width[1] and \
                            lpr_height >= self.plate_signel_height[0] and \
                            lpr_height < self.plate_signel_height[1]:
                            bool_add_lpr = True

                        if bbox_info_idx['column'] == 'Double' and \
                            lpr_width >= self.plate_double_width[0] and \
                            lpr_width < self.plate_double_width[1] and \
                            lpr_height >= self.plate_double_height[0] and \
                            lpr_height < self.plate_double_height[1]:
                            bool_add_lpr = True

                if bool_add_lpr:
                    bbox_state_dict['lpr_num'] += 1
                    bbox_state_dict['lpr_country_list'].append(bbox_info_idx['country'])
                    bbox_state_dict['lpr_city_list'].append(bbox_info_idx['city'])
                    bbox_state_dict['lpr_car_type_list'].append(bbox_info_idx['car_type'])
                    bbox_state_dict['lpr_kind_list'].append(bbox_info_idx['kind'])
                    bbox_state_dict['lpr_num_list'].append(bbox_info_idx['num'])
                    bbox_state_dict['lpr_column_list'].append(bbox_info_idx['column'])
                    bbox_state_dict['lpr_score_list'].append(bbox_info_idx['score'])

                self.params_dict['bbox_state_container'][bbox_state_dict['id']] = bbox_state_dict

                # 信息同步
                bbox_info_idx['frame_num'] = bbox_state_dict['frame_num']
                bbox_info_idx['lpr_num'] = bbox_state_dict['lpr_num']

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
        for key, capture_dict_idy in self.params_dict['capture_container'].items():
          
            # pop
            if capture_dict_idy['capture_frame_num'] > self.capture_frame_num_threshold:
                pop_key_list.append(key)
                        
            capture_dict_idy['capture_frame_num'] += 1
        
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
            report_flage = False

            loc_center_x = (bbox_state_idy['stable_loc'][0] + bbox_state_idy['stable_loc'][2]) / 2.0
            loc_center_y = (bbox_state_idy['stable_loc'][1] + bbox_state_idy['stable_loc'][3]) / 2.0

            # 如果车辆向近处行驶, bbox_state_idy['up_down_state_frame_num'] >= 3 条件用于避免刚进 ROI 或者车辆静止状态下的误判
            if bbox_state_idy['up_down_state'] == 'Near' and bbox_state_idy['up_down_state_frame_num'] >= 3:
                if abs(loc_center_y - self.Down_threshold) < self.capture_plate_up_down_distance_boundary_threshold and \
                    bbox_state_idy['lpr_num'] > self.capture_plate_frame_threshold:
                    near_flage = True

            # 如果车辆向远处行驶，bbox_state_idy['up_down_state_frame_num'] 条件用于避免刚进 ROI 或者车辆静止状态下的误判
            if bbox_state_idy['up_down_state'] == 'Far' and bbox_state_idy['up_down_state_frame_num'] >= 3:
                if abs(loc_center_y - self.Up_threshold) < self.capture_plate_up_down_distance_boundary_threshold and \
                    bbox_state_idy['lpr_num'] > self.capture_plate_frame_threshold:
                    far_flage = True

            # 如果车辆向左边行驶
            if bbox_state_idy['left_right_state'] == 'Left' and bbox_state_idy['left_right_state_frame_num'] >= 3:
                if (( loc_center_x - self.Left_threshold > 0 and \
                      loc_center_x - self.Left_threshold < self.capture_plate_left_right_distance_near_boundary_threshold ) or \
                    ( self.Left_threshold - loc_center_x > 0 and \
                      self.Left_threshold - loc_center_x < self.capture_plate_left_right_distance_far_boundary_threshold )) and \
                    bbox_state_idy['lpr_num'] > self.capture_plate_frame_threshold:
                    left_flage = True
            
            # 如果车辆向右边行驶
            if bbox_state_idy['left_right_state'] == 'Right' and bbox_state_idy['left_right_state_frame_num'] >= 3:
                if (( loc_center_x - self.Right_threshold > 0 and \
                      loc_center_x - self.Right_threshold < self.capture_plate_left_right_distance_far_boundary_threshold ) or \
                    ( self.Right_threshold - loc_center_x > 0 and \
                      self.Right_threshold - loc_center_x < self.capture_plate_left_right_distance_near_boundary_threshold )) and \
                    bbox_state_idy['lpr_num'] > self.capture_plate_frame_threshold:
                    right_flage = True

            # 如果车辆在视野内，超过 25 帧
            if bbox_state_idy['lpr_num'] >= self.capture_outtime_frame_threshold_01:
                outtime_flage_01 = True

            # 更新 capture_dict 抓拍字典
            capture_dict = {}                                                   # 抓拍
            capture_dict['id'] = bbox_state_idy['id']                           # 抓拍id
            capture_dict['flage'] = ''                                          # 抓拍标志信息
            capture_dict['capture_frame_num'] = 0                               # 抓拍帧数
            capture_dict['capture_bool'] = False                                # 抓拍成功标志

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

            if report_flage:

                # 更新 capture_dict 抓拍字典
                if bbox_state_idy['id'] not in self.params_dict['capture_container']:
                    
                    self.params_dict['capture_container'][capture_dict['id']] = capture_dict                

        return


    def update_capture_state(self):

        # update
        pop_key_list = []
        for key, capture_res_dict_idy in self.params_dict['capture_res_container'].items():
          
            # pop
            if capture_res_dict_idy['capture_frame_num'] > self.capture_clear_frame_num_threshold:
                pop_key_list.append(key)
            
            capture_res_dict_idy['capture_frame_num'] += 1
        
        # pop
        for idx in range(len(pop_key_list)):
            self.params_dict['capture_res_container'].pop(pop_key_list[idx])

        # 抓拍逻辑
        # 1、查找稳定结果
        for capture_id_idx, capture_dict_idy in self.params_dict['capture_container'].items():
            
            if capture_dict_idy['capture_bool']:
                continue
    
            # capture_res_dict
            capture_res_dict = {}
            capture_res_dict['id'] = 0                                          # 抓拍id
            capture_res_dict['country'] = 'None'                                # 车牌国家结果
            capture_res_dict['city'] = 'None'                                   # 车牌城市结果
            capture_res_dict['car_type'] = 'None'                               # 车牌车型结果
            capture_res_dict['kind'] = 'None'                                   # 车牌编号结果
            capture_res_dict['num'] = 'None'                                    # 车牌号码结果
            capture_res_dict['column'] = 'None'                                 # 车牌单双行
            capture_res_dict['flage'] = ''                                      # 抓拍标志信息
            capture_res_dict['capture_frame_num'] = 0                           # 抓拍帧数
            capture_res_dict['img_bbox_info_list'] = []                         # 抓拍结果
            capture_res_dict['draw_bool'] = False                               # 绘图标志

            for _, bbox_state_idy in self.params_dict['bbox_state_container'].items():

                if bbox_state_idy['id'] == capture_id_idx:

                    lpr_country_np = np.array(bbox_state_idy['lpr_country_list'])
                    lpr_country_np = lpr_country_np[lpr_country_np != "None"]
                    lpr_city_np = np.array(bbox_state_idy['lpr_city_list'])
                    lpr_city_np = lpr_city_np[lpr_city_np != "None"]
                    lpr_car_type_np = np.array(bbox_state_idy['lpr_car_type_list'])
                    lpr_car_type_np = lpr_car_type_np[lpr_car_type_np != "None"]
                    lpr_kind_np = np.array(bbox_state_idy['lpr_kind_list'])
                    lpr_num_np = np.array(bbox_state_idy['lpr_num_list'])
                    lpr_column_np = np.array(bbox_state_idy['lpr_column_list'])
                    lpr_score_np = np.array(bbox_state_idy['lpr_score_list'])

                    # 获得抓拍序列
                    if len(lpr_num_np[lpr_score_np > self.capture_plate_ocr_score_threshold]):
                        capture_lpr_kind, capture_lpr_kind_frame = Counter(list(lpr_kind_np[lpr_score_np > self.capture_plate_ocr_score_threshold])).most_common(1)[0]
                        capture_lpr_num, capture_lpr_num_frame = Counter(list(lpr_num_np[lpr_score_np > self.capture_plate_ocr_score_threshold])).most_common(1)[0]
                        capture_lpr_column, capture_lpr_column_frame = Counter(list(lpr_column_np)).most_common(1)[0]
                        
                        if capture_lpr_kind_frame >= self.capture_lpr_kind_frame_threshold and \
                            capture_lpr_num_frame >= self.capture_lpr_num_frame_threshold:
                            capture_from_container_list = self.find_capture_plate(bbox_state_idy['id'], capture_lpr_num)
                        
                            # 抓拍车牌
                            if len(capture_from_container_list):

                                capture_dict_idy['capture_bool'] = True
                                capture_res_dict['id'] = capture_id_idx

                                if lpr_country_np.shape[0]:
                                    capture_lpr_country, capture_lpr_country_frame = Counter(list(lpr_country_np)).most_common(1)[0]
                                    if capture_lpr_country_frame >= self.capture_lpr_contry_frame_threshold:
                                        capture_res_dict['country'] = capture_lpr_country
                                    else:
                                        capture_res_dict['country'] = "None"
                                else:
                                    capture_res_dict['country'] = "None"
                                
                                if lpr_city_np.shape[0]:
                                    capture_lpr_city, capture_lpr_city_frame = Counter(list(lpr_city_np)).most_common(1)[0]
                                    if capture_lpr_city_frame >= self.capture_lpr_city_frame_threshold:
                                        capture_res_dict['city'] = capture_lpr_city
                                    else:
                                        capture_res_dict['city'] = "None"
                                else:
                                    capture_res_dict['city'] = "None"
                                
                                if lpr_car_type_np.shape[0]:
                                    capture_lpr_car_type, capture_lpr_car_type_frame = Counter(list(lpr_car_type_np)).most_common(1)[0]
                                    if capture_lpr_car_type_frame >= self.capture_lpr_car_type_frame_threshold:
                                        capture_res_dict['car_type'] = capture_lpr_car_type
                                    else:
                                        capture_res_dict['car_type'] = "None"
                                else:
                                    capture_res_dict['car_type'] = "None"

                                capture_res_dict['kind'] = capture_lpr_kind
                                capture_res_dict['num'] = capture_lpr_num
                                capture_res_dict['column'] = capture_lpr_column
                                capture_res_dict['flage'] = capture_dict_idy['flage']
                                capture_res_dict['img_bbox_info_list'] = capture_from_container_list

                                if capture_res_dict['num'] not in self.params_dict['capture_res_container']:
    
                                    self.params_dict['capture_res_container'][capture_res_dict['num']] = capture_res_dict

        return 


    def find_capture_plate(self, captute_id, capture_license_palte):
        
        capture_from_container_list = []

        for idy in range(len(self.params_dict['bbox_info_container'])):
            bbox_info_list = self.params_dict['bbox_info_container'][idy]['bbox_info']

            for idx in range(len(bbox_info_list)):
                bbox_info_idx = bbox_info_list[idx]

                # 容器中存在追踪对象
                if bbox_info_idx['id'] == captute_id and bbox_info_idx['num'] == capture_license_palte:
                    capture_from_container_list.append(self.params_dict['bbox_info_container'][idy])
        
        if len(capture_from_container_list) > 3:
            capture_from_container_list = random.sample(capture_from_container_list, 3)
        
        return capture_from_container_list