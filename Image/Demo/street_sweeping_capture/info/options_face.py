from easydict import EasyDict as edict

face = edict()

###########################################
# detector
###########################################
face = edict()

# ssd
# 1class_wider_face
# pytorch 
face.ssd_prototxt = None
face.ssd_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2023-05-12-08_focalloss_1class_wider_face/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
# caffe
# face.ssd_prototxt = ""
# face.ssd_model_path = ""
# openvino
# lpr.ssd_prototxt = None
# lpr.ssd_model_path = ""

# # yolov6
# face.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_zg_face_20230515/yolov6_zg_face.py"
# face.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_zg_face_20230515/epoch_300.pth"
# face.yolov6_class_name = ['face']
# face.yolov6_threshold_list = [0.4]
# yolov6 landmark
face.landmark_bool = True
face.landmark_degree_bool = False
face.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_20230526/yolov6_face_wider_face.py"
face.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_20230526/epoch_400.pth"
face.yolov6_class_name = ['face']
face.yolov6_threshold_list = [0.4]
# yolov6 landmark center offset
# face.landmark_bool = True
# face.landmark_degree_bool = False
# face.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_center_offset_20230525/yolov6_face_wider_face.py"
# face.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_center_offset_20230525/epoch_400.pth"
# face.yolov6_class_name = ['face']
# face.yolov6_threshold_list = [0.4]
# # yolov6 landmark degree
# face.landmark_bool = False
# face.landmark_degree_bool = True
# face.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_degree_wider_face_20230525/yolov6_face_wider_face.py"
# face.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_landmark_degree_wider_face_20230525/epoch_260.pth"
# face.yolov6_class_name = ['face']
# face.yolov6_threshold_list = [0.4]

###########################################
# sort
###########################################
face.max_age = 10
face.min_hits = 3 
face.iou_threshold = 0.1
face.sort_expand_ratio = 1.5
face.sort_class_name = ['face']