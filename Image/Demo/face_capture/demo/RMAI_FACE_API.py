import numpy as np
import sys

from pyrsistent import v 

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Demo.face_capture.face.model.detector import FairMot
from Image.Demo.face_capture.face.tracker.multitracker import JDETracker

class FaceCaptureApi():
    """
    FaceCaptureApi
    """

    def __init__(self):
        # option
        self.option_init()

        # param
        self.param_init()

        # model
        self.model_init()

    def option_init(self):
        self.image_width = 1920
        self.image_height = 1080

        # pytorch
        self.fairmot_pytorch_model_path = "/mnt/huanyuan/model_final/image_model/fairmot_centernet_face_zg/centernet_0607/model_last.pth"

        # 上报时机
        self.update_capture_threshold = 40          # 行人框连续丢失上报
        self.speed_threshold = 5.0                  # 行人上报速度阈值
        
        # 状态容器长度
        self.bbox_state_container_length = 50       # 行人框连续丢失，从容器中清除该行人信息

        # 更新车辆行驶状态
        self.update_state_container_length = 1      # 行人框容器大小，用于判断运动状态
        self.update_state_num_threshold = 3         # 行人运动状态计数最大值，用于记录行人处于同一状态的帧数
        self.update_state_threshold = 5.0
        self.update_state_stable_face_alpha = float(0.7)   # 平滑人脸参数


    def param_init(self):
        self.params_dict = {}

        # bbox_info_dict
        bbox_info_dict = {}
        bbox_info_dict['id'] = 0                                            # 追踪id
        bbox_info_dict['loc'] = []                                          # 行人坐标
        bbox_info_dict['score'] = []                                        # 行人得分
        bbox_info_dict['stable_loc'] = []                                   # 行人坐标（稳定）
        bbox_info_dict['state'] = 'Stop'                                    # 行人状态（上下行）
        bbox_info_dict['state_frame_num'] = 0                               # 行人状态（上下行）帧数
        bbox_info_dict['frame_num'] = 0                                     # 行人进入画面帧数
        bbox_info_dict['speed'] = 0                                         # 行人速度

        # bbox_state_dict
        bbox_state_dict = {}
        bbox_state_dict['id'] = 0                                           # 追踪id
        bbox_state_dict['loc'] = []                                         # 行人坐标
        bbox_state_dict['score'] = []                                       # 行人得分
        bbox_state_dict['loc_list'] = []                                    # 行人坐标（多帧）
        bbox_state_dict['stable_loc'] = []                                  # 行人坐标（稳定）
        bbox_state_dict['state'] = 'Stop'                                   # 行人状态（上下行）
        bbox_state_dict['state_frame_num'] = 0                              # 行人状态（上下行）帧数
        bbox_state_dict['frame_num'] = 0                                    # 行人进入画面帧数
        bbox_state_dict['disappear_frame_num'] = 0                          # 行人消失画面帧数
        bbox_state_dict['speed'] = 0                                        # 行人速度
        bbox_state_dict['speed_list'] = []                                  # 行人速度（多帧）
        bbox_state_dict['center_point_list'] = []                           # 行人中心点轨迹（多帧）
        bbox_state_dict['max_speed'] = 5                                    # 最大行人速度
        bbox_state_dict['max_speed_frame_idx'] = 0                          # 最大行人速度帧数
        bbox_state_dict['max_socre'] = 0                                    # 最大行人得分
        bbox_state_dict['caputure_loc'] = 0                                 # 行人保存坐标
        bbox_state_dict['caputure_img'] = 0                                 # 行人保存图像
        bbox_state_dict['captutr_bool'] = False                             # 行人抓拍标志


        self.params_dict['bbox_state_container'] = {}                       # 状态信息容器（key: 追踪id, value: bbox_state_dict）

        # bbox_capture_dict
        bbox_capture_dict = {}
        bbox_capture_dict['id'] = 0                                         # 追踪id
        bbox_capture_dict['loc'] = []                                       # 行人坐标
        bbox_capture_dict['frame_idx'] = []                                 # 行人抓拍帧率
        bbox_capture_dict['img'] = 0                                        # 行人抓拍图像


    def model_init(self):
        # detector
        self.detector = FairMot(self.fairmot_pytorch_model_path, self.image_width, self.image_height)

        # tracker
        self.tracker = JDETracker()


    def run(self, img, frame_idx):

        # info 
        image_width = img.shape[1]
        image_height = img.shape[0]

        assert image_width == self.image_width
        assert image_height == self.image_height

        # detector
        ''' Step 1: Network forward, get detections & embeddings'''
        dets, id_feature = self.detector.detect( img )
            
        # tracker
        tracker_bboxes = self.tracker.update( dets, id_feature )
        
        # update bbox info
        bbox_info_list = self.update_bbox_info( tracker_bboxes )

        # 更新状态容器，同时更新行人状态和帧率
        bbox_info_list = self.update_bbox_state_container( bbox_info_list, img, frame_idx )
        bbox_state_map = self.params_dict['bbox_state_container']

        # 抓拍策略
        capture_dict = self.update_capture( )

        return tracker_bboxes, bbox_info_list, bbox_state_map, capture_dict

    def end_video( self ):
        # 抓拍策略
        capture_dict = self.update_capture( end_bool=True )

        return None, None, None, capture_dict

    def update_bbox_info(self, tracker_bboxes):
        bbox_info_list = []
        for idx in range(len(tracker_bboxes)):
            # init 
            # bbox_info_dict
            bbox_info_dict = {}
            bbox_info_dict['id'] = 0                                            # 追踪id
            bbox_info_dict['loc'] = []                                          # 行人坐标
            bbox_info_dict['score'] = []                                        # 行人得分
            bbox_info_dict['stable_loc'] = []                                   # 行人坐标（稳定）
            bbox_info_dict['state'] = 'Stop'                                    # 行人状态（上下行）
            bbox_info_dict['state_frame_num'] = 0                               # 行人状态（上下行）帧数
            bbox_info_dict['frame_num'] = 0                                     # 行人进入画面帧数
            bbox_info_dict['speed'] = 0                                         # 行人速度

            tracker_bbox = np.array(tracker_bboxes[idx])
            bbox_info_dict['loc'] = tracker_bbox[0:4]
            bbox_info_dict['score'] = tracker_bbox[4]
            bbox_info_dict['id'] = tracker_bbox[5]

            bbox_info_list.append(bbox_info_dict)

        return bbox_info_list


    def update_bbox_state_container(self, bbox_info_list, img, frame_idx):

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
                    bbox_state_idy['score'] = bbox_info_idx['score']
                    bbox_state_idy['loc_list'].append(bbox_info_idx['loc'])
                    if len(bbox_state_idy['loc_list']) >= self.update_state_container_length: 
                        bbox_state_idy['loc_list'].pop(0)
                    
                    # 更新行人速度
                    new_stable_loc = self.update_state_stable_face_alpha * bbox_state_idy['stable_loc'] + (1 - self.update_state_stable_face_alpha) * bbox_info_idx['loc']
                    stable_center_y = ( bbox_state_idy['stable_loc'][1] + bbox_state_idy['stable_loc'][3] ) / 2
                    new_stable_center_y = ( new_stable_loc[1] + new_stable_loc[3] ) / 2
                    bbox_state_idy['speed'] = (stable_center_y - new_stable_center_y) / float(bbox_state_idy['disappear_frame_num'])

                    # 更新行人状态
                    bbox_state_idy['stable_loc'] = new_stable_loc
                    center_y = ( np.array(bbox_state_idy['loc_list']).mean(0)[1] + np.array(bbox_state_idy['loc_list']).mean(0)[3] ) / 2
                    stable_center_y = ( bbox_state_idy['stable_loc'][1] + bbox_state_idy['stable_loc'][3] ) / 2
                    distance_y = stable_center_y - center_y
                    
                    bbox_state_idy['disappear_frame_num'] = 0

                    # 行人状态判断
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

                    # 保存抓拍信息
                    if bbox_state_idy['speed'] > bbox_state_idy['max_speed']:
                        bbox_state_idy['max_speed'] = bbox_state_idy['speed']
                        bbox_state_idy['max_speed_frame_idx'] = frame_idx
                    if bbox_state_idy['score'] > bbox_state_idy['max_socre'] and bbox_state_idy['state'] == "Up":
                        bbox_state_idy['max_socre'] = bbox_state_idy['score']
                        bbox_state_idy['caputure_loc'] = bbox_state_idy['loc']                              
                        bbox_state_idy['caputure_img'] = img                             

                    bbox_info_idx['state'] = bbox_state_idy['state']
                    bbox_info_idx['state_frame_num'] = bbox_state_idy['state_frame_num']
                    bbox_info_idx['stable_loc'] = bbox_state_idy['stable_loc']
                    bbox_info_idx['frame_num'] = bbox_state_idy['frame_num']
                    bbox_info_idx['speed'] = bbox_state_idy['speed']
                        
            if is_new_id_bool:

                # bbox_state_dict
                bbox_state_dict = {}
                bbox_state_dict['id'] = 0                                           # 追踪id
                bbox_state_dict['loc'] = []                                         # 行人坐标
                bbox_state_dict['score'] = []                                       # 行人得分
                bbox_state_dict['loc_list'] = []                                    # 行人坐标（多帧）
                bbox_state_dict['stable_loc'] = []                                  # 行人坐标（稳定）
                bbox_state_dict['state'] = 'Stop'                                   # 行人状态（上下行）
                bbox_state_dict['state_frame_num'] = 0                              # 行人状态（上下行）帧数
                bbox_state_dict['frame_num'] = 0                                    # 行人进入画面帧数
                bbox_state_dict['disappear_frame_num'] = 0                          # 行人消失画面帧数
                bbox_state_dict['speed'] = 0                                        # 行人速度
                bbox_state_dict['speed_list'] = []                                  # 行人速度（多帧）
                bbox_state_dict['center_point_list'] = []                           # 行人中心点轨迹（多帧）
                bbox_state_dict['max_speed'] = 5                                    # 最大行人速度
                bbox_state_dict['max_speed_frame_idx'] = 0                          # 最大行人速度帧数
                bbox_state_dict['max_socre'] = 0                                    # 最大行人得分
                bbox_state_dict['caputure_loc'] = 0                                 # 行人保存坐标
                bbox_state_dict['caputure_img'] = 0                                 # 行人保存图像
                bbox_state_dict['captutr_bool'] = False                             # 行人抓拍标志

                bbox_state_dict['id'] = bbox_info_idx['id']
                bbox_state_dict['loc'] = bbox_info_idx['loc']
                bbox_state_dict['score'] = bbox_info_idx['score']
                bbox_state_dict['loc_list'].append(bbox_info_idx['loc'])
                bbox_state_dict['stable_loc'] = bbox_info_idx['loc']
                bbox_info_idx['stable_loc'] = bbox_info_idx['loc']

                self.params_dict['bbox_state_container'][bbox_state_dict['id']] = bbox_state_dict

        # update
        pop_key_list = []
        for key, bbox_state_idy in self.params_dict['bbox_state_container'].items():
            
            if bbox_state_idy['disappear_frame_num'] > 0:
                bbox_state_idy['speed_list'].append( 0.0 )
            else:

                bbox_state_idy['speed_list'].append( bbox_state_idy['speed'] )
            bbox_state_idy['center_point_list'].append( ( (bbox_state_idy['loc'][0] + bbox_state_idy['loc'][2]) / 2 , (bbox_state_idy['loc'][1] + bbox_state_idy['loc'][3]) / 2 ) )
            
            if len( bbox_state_idy['speed_list'] ) > self.bbox_state_container_length:
                bbox_state_idy['speed_list'].pop(0)
                bbox_state_idy['center_point_list'].pop(0)

        return bbox_info_list
    
    def update_capture( self, end_bool=False ):

        capture_dict = {}

        for key, bbox_state_idy in self.params_dict['bbox_state_container'].items():
            
            if ( end_bool and \
                    bbox_state_idy['max_speed'] > self.speed_threshold and \
                    not bbox_state_idy['captutr_bool'] ) or \
                ( not end_bool and \
                    bbox_state_idy['disappear_frame_num'] > self.update_capture_threshold and \
                    bbox_state_idy['max_speed'] > self.speed_threshold and \
                    not bbox_state_idy['captutr_bool']) :

                # bbox_capture_dict
                bbox_capture_dict = {}
                bbox_capture_dict['id'] = 0                                         # 追踪id
                bbox_capture_dict['loc'] = []                                       # 行人坐标
                bbox_capture_dict['frame_idx'] = []                                 # 行人抓拍帧率
                bbox_capture_dict['img'] = 0                                        # 行人抓拍图像

                bbox_capture_dict['id'] = bbox_state_idy['id']
                bbox_capture_dict['loc'] = bbox_state_idy['caputure_loc']
                bbox_capture_dict['frame_idx'] = bbox_state_idy['max_speed_frame_idx']
                bbox_capture_dict['img'] = bbox_state_idy['caputure_img']

                bbox_state_idy['captutr_bool'] = True

                capture_dict[bbox_capture_dict['id']] = bbox_capture_dict

        return capture_dict
