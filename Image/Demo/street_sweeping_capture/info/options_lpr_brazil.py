from easydict import EasyDict as edict

options = edict()


###########################################
# resolution
###########################################
# # 5M
# options.image_width = 2592
# options.image_height = 1920

# 2M
options.image_width = 1920
options.image_height = 1080

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
# lpr
options.ssd_bool = False
options.yolov6_bool =True
options.ssd_caffe_bool = False
options.ssd_openvino_bool = False

# # ssd
# # zd_ssd_rfb_wmr
# # pytorch 
# # options.ssd_prototxt = None
# # options.ssd_model_path = ""
# # caffe
# options.ssd_prototxt = "/mnt/huanyuan/model_final/image_model/schoolbus/zd_ssd_rfb_wmr/ssd_mbv2_2class/caffe_model/ssd_mobilenetv2_fpn.prototxt"
# options.ssd_model_path = "/mnt/huanyuan/model_final/image_model/schoolbus/zd_ssd_rfb_wmr/ssd_mbv2_2class/caffe_model/ssd_mobilenetv2_0421.caffemodel"
# # openvino
# # options.ssd_prototxt = None
# # options.ssd_model_path = ""
# options.ssd_class_name = ['license_plate']
# options.ssd_conf_thres = 0.25

# # SSD_VGG_FPN_RFB_2023-06-09_focalloss_5class_car_bus_truck_motorcyclist_licenseplate_softmax
# # pytorch 
# # options.ssd_prototxt = None
# # options.ssd_model_path = ""
# # caffe
# options.ssd_prototxt = "/mnt/huanyuan/model_final/image_model/schoolbus/ssd_rfb/SSD_VGG_FPN_RFB_2023-06-09_focalloss_5class_car_bus_truck_motorcyclist_licenseplate_softmax/FPN_RFB_4class_3attri_noDilation_prior.prototxt"
# options.ssd_model_path = "/mnt/huanyuan/model_final/image_model/schoolbus/ssd_rfb/SSD_VGG_FPN_RFB_2023-06-09_focalloss_5class_car_bus_truck_motorcyclist_licenseplate_softmax/SSD_VGG_FPN_RFB_VOC_car_bus_truck_motorcyclist_licenseplate_2023_06_09_70.caffemodel"
# # openvino
# # options.ssd_prototxt = None
# # options.ssd_model_path = ""

# # yolov6
# options.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_jpf/yolov6.py"
# options.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_jpf/epoch_260.pth"
# options.yolov6_class_name = ['car', 'bus', 'truck', 'car_reg', 'car_big_reg', 'car_front',
#                         'car_big_front', 'person', 'motorcyclist', 'bicyclist',
#                         'sign_upspeed_round', 'sign_upspeed_square', 'sign_stop', 'sign_height',
#                         'light_share0', 'light_share', 'bridge', 'zebra_crossing', 'license_plate']
# options.yolov6_threshold_list = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

# yolov6_c27_car_bus_truck_moto_plate_0731
options.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_c27_car_bus_truck_moto_plate_0731/yolov6_rm_c27_deploy.py"
options.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_c27_car_bus_truck_moto_plate_0731/epoch_300_deploy.pth"
options.yolov6_class_name = ["car", "bus", "truck", "motorcyclist", "license_plate"]
options.yolov6_threshold_list = [0.3, 0.3, 0.3, 0.3, 0.3]

# # 是否将 car\bus\truck 合并为一类输出
options.car_attri_merge_bool = True
# options.car_attri_merge_bool = False
options.car_attri_merge_name = 'car_bus_truck'
options.car_attri_name_list = [ 'car', 'bus', 'truck', 'motorcyclist' ]
options.license_plate_name = 'license_plate'


###########################################
# options
###########################################
options.lpr_caffe_bool = False
options.lpr_pytorch_bool = False
options.lpr_onnx_bool = True

################
# brazil
################
options.brazil = edict()

# # mexico_1201_lxn
# options.brazil.ocr_pth_path = ""
# options.brazil.ocr_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr/lpr_bm_lxn/mexico_1201/cnn_turkey.prototxt"
# options.brazil.ocr_caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr/lpr_bm_lxn/mexico_1201/mexico_1201.caffemodel"
# options.brazil.input_shape = (256, 32)

# lpr single line paddle
options.lpr_paddle_bool = True
options.brazil.ocr_pth_path = ""
options.brazil.ocr_caffe_prototxt = "/mnt/huanyuan/model/image/lpr/paddle_ocr/v1_brazil_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230824_wdiffste_NoAug_202309/inference/caffe/model.prototxt"
options.brazil.ocr_caffe_model_path = "/mnt/huanyuan/model/image/lpr/paddle_ocr/v1_brazil_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230824_wdiffste_NoAug_202309/inference/caffe/model.caffemodel"
options.brazil.ocr_onnx_model_path = "/mnt/huanyuan/model/image/lpr/paddle_ocr/v1_brazil_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230824_wdiffste_NoAug_202309/inference/onnx/model.onnx"
options.brazil.input_shape = (1, 64, 256)
options.brazil.ocr_labels_dict_path = "/mnt/huanyuan/model/image/lpr/paddle_ocr/v1_brazil_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230824_wdiffste_NoAug_202309/inference/brazil_dict.txt"
options.brazil.ocr_prefix_beam_search_bool = False
options.brazil.padding_bool = False

# lpr first line
options.brazil.ocr_first_line_pth_path = "/mnt/huanyuan/model/image/lpr/brazil/ocr_brazil_first_line_20230116/crnn_best.pth"
options.brazil.ocr_first_line_caffe_prototxt = "/mnt/huanyuan/model/image/lpr/brazil/ocr_brazil_first_line_20230116/cnn_256x64_38.prototxt"
options.brazil.ocr_first_line_caffe_model_path = "/mnt/huanyuan/model/image/lpr/brazil/ocr_brazil_first_line_20230116/ocr_brazil_first_line_20230116.caffemodel"
options.brazil.ocr_first_line_caffe_shape = (256, 64)

# lpr second line
options.brazil.ocr_second_line_pth_path = "/mnt/huanyuan/model/image/lpr/brazil/ocr_brazil_second_line_20230116/crnn_best.pth"
options.brazil.ocr_second_line_caffe_prototxt = "/mnt/huanyuan/model/image/lpr/brazil/ocr_brazil_second_line_20230116/cnn_256x64_38.prototxt"
options.brazil.ocr_second_line_caffe_model_path = "/mnt/huanyuan/model/image/lpr/brazil/ocr_brazil_second_line_20230116/ocr_brazil_second_line_20230116.caffemodel"
options.brazil.ocr_second_line_caffe_shape = (256, 64)

options.brazil.ocr_labels = ['-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                                'J', 'K', 'L', 'M', 'N', 'P', 'O', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','#']
options.brazil.ocr_prefix_beam_search_bool = False
options.brazil.padding_bool = False

# 车牌长宽阈值
# options.plate_height = [35, 960]
# options.plate_width = [0, 1920]
options.plate_signel_height = [35, 960]      # 25 -> 30 - 35
options.plate_signel_width = [0, 1920]
options.plate_double_height = [55, 960]      # 45 -> 50 - 55   
options.plate_double_width = [0, 1920]

options.lpr_ocr_width_expand_ratio = 0.05
options.lpr_ocr_column_threshold = 2.0


###########################################
# sort
###########################################
# options.sort_type = "car"
options.sort_type = "plate"

# car
options.max_age = 10
options.min_hits = 3 
options.iou_threshold = 0.1
options.sort_expand_ratio = 1.5
options.sort_class_name = ['license_plate']

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
options.capture_outtime_frame_threshold_double_01 = 7

options.capture_lpr_score_threshold = 0.8
options.capture_lpr_num_frame_threshold = 4
options.capture_lpr_kind_frame_threshold = 4
options.capture_lpr_color_frame_threshold = 3


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