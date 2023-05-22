import sys 

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Demo.bus_face_capture.p3d.mdoel.detector import *
from Image.Demo.bus_face_capture.p3d.tracker.sort import *


class P3DCaptureApi():
    """
    P3DCaptureApi
    """

    def __init__(self):
        # option
        self.option_init()

        # param
        self.param_init()
        
        # model_init 
        self.model_init()


    def option_init(self):
        self.image_width = 1920
        self.image_height = 1080

        # caffe
        self.ssd_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/ssd_rfb_p3d_face_wdx/FPN_RFB_2class_prior_noRFB.prototxt"
        self.ssd_caffe_model_path = "/mnt/huanyuan/model_final/image_model/ssd_rfb_p3d_face_wdx/ssd_0525_2cls_hisi.caffemodel"
        self.track_class_name = 'person'

        # sort
        self.max_age = 10
        self.min_hits = 3 
        self.iou_threshold = 0.3

        # roi 标定框
        # self.roi_area = [1094, 910, 1412, 1010]
        # self.roi_area = [750, 600, 1160, 670]
        self.roi_area = [920, 600, 1200, 720]

        # roi 标定框 人脸区域
        # self.roi_area_person = [ 940, 480, 1570, 1070]
        # self.roi_area_person = [ 730, 330, 1220, 810]
        self.roi_area_person = [ 880, 270, 1240, 760]

        # 状态容器长度
        self.bbox_state_container_length = 50       # 行人框连续丢失上报，从容器中清除该行人信息


    def param_init(self):
        self.params_dict = {}

        # bbox_info_dict
        bbox_info_dict = {}
        bbox_info_dict['id'] = 0                                            # 追踪id
        bbox_info_dict['loc'] = []                                          # 行人坐标
        bbox_info_dict['frame_num'] = 0                                     # 行人进入画面帧数
        bbox_info_dict['captue_bool'] = False                               # 行人抓拍标注位

        # bbox_state_dict
        bbox_state_dict = {}
        bbox_state_dict['id'] = 0                                           # 追踪id
        bbox_state_dict['loc'] = []                                         # 行人坐标
        bbox_state_dict['frame_num'] = 0                                    # 行人进入画面帧数
        bbox_state_dict['disappear_frame_num'] = 0                          # 行人消失画面帧数
        bbox_state_dict['captue_bool'] = False                              # 行人抓拍标注位
        bbox_state_dict['captue_bool_list'] = [ ]                           # 行人抓拍标注位（多帧）

        self.params_dict['bbox_state_container'] = {}                       # 状态信息容器（key: 追踪id, value: bbox_state_dict）

        # bbox_capture_dict
        bbox_capture_dict = {}
        bbox_capture_dict['id'] = 0                                         # 追踪id
        bbox_capture_dict['loc'] = []                                       # 行人坐标
        bbox_capture_dict['frame_idx'] = []                                 # 行人抓拍帧率
        bbox_capture_dict['img'] = 0                                        # 行人抓拍图像


    def model_init(self):
        # detector
        self.detector = SSDDetector(self.ssd_caffe_prototxt, self.ssd_caffe_model_path)

        # tracker
        self.tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_threshold)


    def run(self, img, frame_idx):
    
        # info 
        image_width = img.shape[1]
        image_height = img.shape[0]

        assert image_width == self.image_width
        assert image_height == self.image_height

        # detector
        bboxes = self.detector.detect( img )

        # tracker
        tracker_bboxes = self.update_tracker_bboxes( bboxes )

        # update bbox info
        bbox_info_list = self.update_bbox_info( tracker_bboxes )
        
        # 更新状态容器，同时更新行人状态和帧率
        bbox_info_list = self.update_bbox_state_container( bbox_info_list )
        bbox_state_map = self.params_dict['bbox_state_container']

        # 抓拍策略
        capture_dict = self.update_capture( bbox_info_list, img, frame_idx )

        return tracker_bboxes, bbox_info_list, bbox_state_map, capture_dict, self.roi_area

    def end_video( self ):

        return None, None, None, {}, None

    def update_tracker_bboxes(self, bboxes):

        if self.track_class_name in bboxes:
            dets = np.array(bboxes[self.track_class_name])
        else:
            dets = np.empty((0, 5))
                
        tracker_bboxes = self.tracker.update(dets)

        return tracker_bboxes


    def update_bbox_info(self, tracker_bboxes):

        bbox_info_list = []
        for idx in range(len(tracker_bboxes)):
            # init 
            # bbox_info_dict
            bbox_info_dict = {}
            bbox_info_dict['id'] = 0                                            # 追踪id
            bbox_info_dict['loc'] = []                                          # 行人坐标
            bbox_info_dict['frame_num'] = 0                                     # 行人进入画面帧数
            bbox_info_dict['captue_bool'] = False                               # 行人抓拍标注位
        
            tracker_bbox = np.array(tracker_bboxes[idx])
            bbox_info_dict['id'] = tracker_bbox[-1]
            bbox_info_dict['loc'] = tracker_bbox[0:4]

            bbox_info_list.append(bbox_info_dict)
        
        return bbox_info_list
    

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
                    bbox_state_idy['loc'] = bbox_info_idx['loc']
                    bbox_state_idy['frame_num'] += 1
                    bbox_state_idy['disappear_frame_num'] = 0

                    center_x = ( np.array(bbox_state_idy['loc'])[0] + np.array(bbox_state_idy['loc'])[2] ) / 2
                    center_y = ( np.array(bbox_state_idy['loc'])[1] + np.array(bbox_state_idy['loc'])[3] ) / 2

                    # 行人抓拍判断
                    capture_bool = False
                    if center_y < self.roi_area[1] and \
                        (center_x > self.roi_area_person[0] and center_x < self.roi_area_person[2] and center_y > self.roi_area_person[1] and center_y < self.roi_area_person[3] ):
                        capture_bool = True
                    
                    if capture_bool and not bbox_state_idy['captue_bool']:
                        bbox_state_idy['captue_bool'] = True
                        bbox_info_idx['captue_bool'] = True
                    
                    bbox_info_idx['frame_num'] = bbox_state_idy['frame_num']

            if is_new_id_bool:
    
                # bbox_state_dict
                bbox_state_dict = {}
                bbox_state_dict['id'] = 0                                           # 追踪id
                bbox_state_dict['loc'] = []                                         # 行人坐标
                bbox_state_dict['frame_num'] = 0                                    # 行人进入画面帧数
                bbox_state_dict['disappear_frame_num'] = 0                          # 行人消失画面帧数
                bbox_state_dict['captue_bool'] = False                              # 行人抓拍标注位
                bbox_state_dict['captue_bool_list'] = [ ]                           # 行人抓拍标注位（多帧）

                bbox_state_dict['id'] = bbox_info_idx['id']
                bbox_state_dict['loc'] = bbox_info_idx['loc']

                self.params_dict['bbox_state_container'][bbox_state_dict['id']] = bbox_state_dict

        # update
        pop_key_list = []
        for key, bbox_state_idy in self.params_dict['bbox_state_container'].items():
            
            bbox_state_idy['captue_bool_list'].append( bbox_state_idy['captue_bool'] )
            
            if len( bbox_state_idy['captue_bool_list'] ) > self.bbox_state_container_length:
                bbox_state_idy['captue_bool_list'].pop(0)

        return bbox_info_list

    
    def update_capture( self, bbox_info_list, img, frame_idx):

        capture_dict = {}

        for idx in range(len(bbox_info_list)):
            bbox_info_idx = bbox_info_list[idx]

            if bbox_info_idx['captue_bool']:

                # bbox_capture_dict
                bbox_capture_dict = {}
                bbox_capture_dict['id'] = 0                                         # 追踪id
                bbox_capture_dict['loc'] = []                                       # 行人坐标
                bbox_capture_dict['frame_idx'] = []                                 # 行人抓拍帧率
                bbox_capture_dict['img'] = 0                                        # 行人抓拍图像

                bbox_capture_dict['id'] = bbox_info_idx['id']
                bbox_capture_dict['loc'] = bbox_info_idx['loc']
                bbox_capture_dict['frame_idx'] = frame_idx
                bbox_capture_dict['img'] = img

                capture_dict[bbox_capture_dict['id']] = bbox_capture_dict
        
        return capture_dict

