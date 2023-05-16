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

# yolov6
face.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_zg_face_20230515/yolov6_zg_face.py"
face.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_zg_face_20230515/epoch_300.pth"
face.yolov6_class_name = ['face']
face.yolov6_threshold_list = [0.4]



###########################################
# sort
###########################################
face.max_age = 10
face.min_hits = 3 
face.iou_threshold = 0.1
face.sort_expand_ratio = 1.5
face.sort_class_name = ['face']