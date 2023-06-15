from easydict import EasyDict as edict

options = edict()

###########################################
# resolution
###########################################
# 5M
options.image_width = 2592
options.image_height = 1920

# other
# options.image_width = 2592
# options.image_height = 1520
# options.image_width = 2688
# options.image_height = 1520

# # 2M
# options.image_width = 1920
# options.image_height = 1080

# # 720p
# options.image_width = 1280
# options.image_height = 720


###########################################
# gpu
###########################################
options.gpu_bool = True
# options.gpu_bool = False
# options.device = 'cpu'
options.device = 'cuda:0'

###########################################
# detector
###########################################
# face
options.ssd_bool = False
options.yolov6_bool = True
options.ssd_caffe_bool = False
options.ssd_openvino_bool = False

# # ssd
# # 1class_wider_face
# # pytorch 
# options.ssd_prototxt = None
# options.ssd_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2023-05-12-08_focalloss_1class_wider_face/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
# # caffe
# # options.ssd_prototxt = ""
# # options.ssd_model_path = ""
# # openvino
# # lpr.ssd_prototxt = None
# # lpr.ssd_model_path = ""

# # yolov6
# options.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_zg_face_20230515/yolov6_zg_face.py"
# options.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_zg_face_20230515/epoch_300.pth"
# options.yolov6_class_name = ['face']
# options.yolov6_threshold_list = [0.4]
# # yolov6 landmark
# options.landmark_bool = True
# options.landmark_degree_bool = False
# options.landmark_degree_cls_bool = False
# options.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_20230526/yolov6_face_wider_face.py"
# options.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_20230526/epoch_400.pth"
# options.yolov6_class_name = ['face']
# options.yolov6_threshold_list = [0.4]
# # yolov6 landmark center offset
# options.landmark_bool = True
# options.landmark_degree_bool = False
# options.landmark_degree_cls_bool = False
# options.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_center_offset_20230525/yolov6_face_wider_face.py"
# options.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_center_offset_20230525/epoch_400.pth"
# options.yolov6_class_name = ['face']
# options.yolov6_threshold_list = [0.4]
# # yolov6 landmark sigmoid center offset qa
# options.landmark_bool = True
# options.landmark_degree_bool = False
# options.landmark_degree_cls_bool = False
# options.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_qa_wider_face_20230605/yolov6_qa_face_wider_face.py"
# options.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_qa_wider_face_20230605/epoch_400.pth"
# options.yolov6_class_name = ['face']
# options.yolov6_threshold_list = [0.4]
# yolox landmark sigmoid center offset
# options.landmark_bool = True
# options.landmark_degree_bool = False
# options.landmark_degree_cls_bool = False
# options.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolox_landmark_wider_face_20230612/yolox_face_wider_face.py"
# options.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolox_landmark_wider_face_20230612/epoch_300.pth"
# options.yolov6_class_name = ['face']
# options.yolov6_threshold_list = [0.4]
# yolov6 landmark degree cls
options.landmark_bool = False
options.landmark_degree_bool = False
options.landmark_degree_cls_bool = True
options.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_degree_wider_face_20230607/yolov6_face_wider_face.py"
options.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_degree_wider_face_20230607/epoch_300.pth"
options.yolov6_class_name = ['face']
options.yolov6_threshold_list = [0.4]

###########################################
# sort
###########################################
options.max_age = 10
options.min_hits = 3 
options.iou_threshold = 0.1
options.sort_expand_ratio = 1.5
options.sort_class_name = ['face']


###########################################
# cache
###########################################
# 缓存间隔
options.cache_interval = 2
# 缓存容器长度
options.cache_container_length = 8


###########################################
# state
###########################################
# 状态容器长度
options.bbox_state_container_length = 10       # 车辆框连续丢失上报，从容器中清除该车辆信息
options.lpr_ocr_state_container_length = 20    # 车牌状态长度阈值
options.lpr_city_state_container_length = 10   # 车牌状态长度阈值
options.lpr_face_state_container_length = 10   # 人脸状态长度阈值

# 更新车辆行驶状态
options.update_state_num_threshold = 5         # 车辆行驶状态计数最大值，用于记录车辆处于同一行驶状态的帧数
options.update_state_threshold = 1
options.update_state_stable_loc_alpha = float(0.6)   # 平滑车辆框参数


###########################################
# capture
###########################################
# 抓拍线
options.capture_line_up_down_ratio = [0.03, 0.5, 0.9, 0.97]
options.capture_line_left_right_ratio = [0.03, 0.25, 0.75, 0.97]

# 报警时间长短
options.capture_frame_num_threshold = 16
options.capture_clear_frame_num_threshold = 60 * 25        # 经过多少帧，抓拍容器清空

options.capture_info_frame_threshold = 5
options.capture_outtime_frame_threshold_01 = 25
options.capture_outtime_frame_threshold_02 = 150
options.capture_up_down_distance_boundary_threshold = 100
options.capture_left_right_distance_near_boundary_threshold = 200
options.capture_left_right_distance_far_boundary_threshold = 400

options.capture_face_landmark_degree_threshold = 0.5
options.face_landmark_degree_frame_threshold  = 4


###########################################
# roi
###########################################

# 是否通过 roi 区域屏蔽部分检测结果
options.roi_bool = False
# options.roi_bool = True
options.roi_area = [0, 0, options.image_width, options.image_height]

# 上下限阈值 & 左右限阈值
if options.roi_bool:
    options.ROI_Up_threshold = options.roi_area[1] + ( options.roi_area[3] - options.roi_area[1] ) * options.capture_line_up_down_ratio[0]
    options.ROI_Down_threshold = options.roi_area[1] + ( options.roi_area[3] - options.roi_area[1] ) * options.capture_line_up_down_ratio[3]
    options.Up_threshold = options.roi_area[1] + ( options.roi_area[3] - options.roi_area[1] ) * options.capture_line_up_down_ratio[1]
    options.Down_threshold = options.roi_area[1] + ( options.roi_area[3] - options.roi_area[1] ) * options.capture_line_up_down_ratio[2]
    options.ROI_Left_threshold = options.roi_area[0] + ( options.roi_area[2] - options.roi_area[0] ) * options.capture_line_left_right_ratio[0]
    options.ROI_Right_threshold = options.roi_area[0] + ( options.roi_area[2] - options.roi_area[0] ) * options.capture_line_left_right_ratio[3]
    options.Left_threshold = options.roi_area[0] + ( options.roi_area[2] - options.roi_area[0] ) * options.capture_line_left_right_ratio[1]
    options.Right_threshold = options.roi_area[0] + ( options.roi_area[2] - options.roi_area[0] ) * options.capture_line_left_right_ratio[2]
else:
    options.ROI_Up_threshold = options.image_height * options.capture_line_up_down_ratio[0]
    options.ROI_Down_threshold = options.image_height * options.capture_line_up_down_ratio[3]
    options.Up_threshold = options.image_height * options.capture_line_up_down_ratio[1]
    options.Down_threshold = options.image_height * options.capture_line_up_down_ratio[2]
    options.ROI_Left_threshold = options.image_width * options.capture_line_left_right_ratio[0]
    options.ROI_Right_threshold = options.image_width * options.capture_line_left_right_ratio[3]
    options.Left_threshold = options.image_width * options.capture_line_left_right_ratio[1]
    options.Right_threshold = options.image_width * options.capture_line_left_right_ratio[2]