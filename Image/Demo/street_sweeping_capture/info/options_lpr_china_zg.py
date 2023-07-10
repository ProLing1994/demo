from easydict import EasyDict as edict

options = edict()


###########################################
# resolution
###########################################
# 5M
options.image_width = 2592
options.image_height = 1920

# # other
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
# options.gpu_bool = True
options.gpu_bool = False
# options.device = 'cpu'
options.device = 'cuda:0'


###########################################
# detector
###########################################
# lpr
options.ssd_bool = True
options.yolov6_bool =False
options.ssd_caffe_bool = True
options.ssd_openvino_bool = False

# ssd
# 2022-07-22-00
# # pytorch 
# options.ssd_prototxt = None
# options.ssd_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-07-22-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
# caffe
# options.ssd_prototxt = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-07-22-00/FPN_RFB_3class_3attri_noDilation_prior.prototxt"
# options.ssd_model_path = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-07-22-00/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-07-22-00.caffemodel"
# # openvino
# options.ssd_prototxt = None
# options.ssd_model_path = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-07-22-00/openvino_model/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-07-22-00.xml"

# # # 2022-08-10-00
# # # pytorch 
# # options.ssd_prototxt = None
# # options.ssd_model_path = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-08-10-00/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
# # # caffe
# options.ssd_prototxt = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-08-10-00/FPN_RFB_3class_3attri_noDilation_prior.prototxt"
# options.ssd_model_path = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-08-10-00/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-08-10-00.caffemodel"
# # # openvino
# # options.ssd_prototxt = None
# # options.ssd_model_path = ""

# SSD_VGG_FPN_RFB_2023-06-09_focalloss_5class_car_bus_truck_motorcyclist_licenseplate_softmax
# pytorch 
# options.ssd_prototxt = None
# options.ssd_model_path = ""
# caffe
options.ssd_prototxt = "/mnt/huanyuan/model_final/image_model/schoolbus/ssd_rfb/SSD_VGG_FPN_RFB_2023-06-09_focalloss_5class_car_bus_truck_motorcyclist_licenseplate_softmax/FPN_RFB_4class_3attri_noDilation_prior.prototxt"
options.ssd_model_path = "/mnt/huanyuan/model_final/image_model/schoolbus/ssd_rfb/SSD_VGG_FPN_RFB_2023-06-09_focalloss_5class_car_bus_truck_motorcyclist_licenseplate_softmax/SSD_VGG_FPN_RFB_VOC_car_bus_truck_motorcyclist_licenseplate_2023_06_09_70.caffemodel"
# openvino
# options.ssd_prototxt = None
# options.ssd_model_path = ""

# yolov6
options.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_jpf/yolov6.py"
options.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_jpf/epoch_260.pth"
options.yolov6_class_name = ['car', 'bus', 'truck', 'car_reg', 'car_big_reg', 'car_front',
                        'car_big_front', 'person', 'motorcyclist', 'bicyclist',
                        'sign_upspeed_round', 'sign_upspeed_square', 'sign_stop', 'sign_height',
                        'light_share0', 'light_share', 'bridge', 'zebra_crossing', 'license_plate']
options.yolov6_threshold_list = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

# 是否将 car\bus\truck 合并为一类输出
# options.car_attri_merge_bool = True
options.car_attri_merge_bool = False
options.car_attri_merge_name = 'car_bus_truck'
options.car_attri_name_list = [ 'car', 'bus', 'truck', 'motorcyclist' ]
options.license_plate_name = 'license_plate'


###########################################
# options
###########################################
options.lpr_caffe_bool = True
options.lpr_pytorch_bool = False

################
# china
################
options.china = edict()
# 0628
# options.  = False
# options.china.ocr_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/0628/china_double_softmax.prototxt"
# options.china.ocr_caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/0628/china_double.caffemodel"
# options.china.input_shape = (256, 64)
# options.china.ocr_labels = ["-","皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
#                             "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
#                             "陕", "甘", "青", "宁", "新", "警", "学", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
#                             'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
#                             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '挂']
# options.china.ocr_prefix_beam_search_bool = False
# options.china.padding_bool = False

# # ocr_cn_20230519
# options.china.ocr_pth_path = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/ocr_cn_20230519/crnn_best.pth"
# options.china.ocr_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/ocr_cn_20230519/cnn_256x64_73.prototxt"
# options.china.ocr_caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/ocr_cn_20230519/ocr_cn_20230519.caffemodel"
# options.china.input_shape = (256, 64)
# options.china.ocr_labels = [ "-", 
#                         "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
#                         "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
#                         "陕", "甘", "青", "宁", "新", "警", "学", '挂', '领', '空', '港', '澳', 
#                         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 
#                         'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
#                         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ]
# options.china.ocr_prefix_beam_search_bool = False
# options.china.padding_bool = True

# paddle_ocr_20230530_cn
options.lpr_paddle_bool = True
options.china.ocr_pth_path = ""
options.china.ocr_caffe_prototxt = "/mnt/huanyuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230530_cn/inference/caffe/model.prototxt"
options.china.ocr_caffe_model_path = "/mnt/huanyuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230530_cn/inference/caffe/model.caffemodel"
options.china.input_shape = (1, 64, 256)
options.china.ocr_labels_dict_path = "/mnt/huanyuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230530_cn/inference/cn_dict.txt"
options.china.ocr_prefix_beam_search_bool = False
options.china.padding_bool = False

# color
# # seg_color_cn_20230512
# options.china.seg_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/seg_color_cn_20230512/seg_color_cn_20230512.prototxt"
# options.china.seg_caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/seg_color_cn_20230512/seg_color_cn_20230512.caffemodel"
# seg_color_cn_20230516
# options.china.seg_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/seg_color_cn_20230516/seg_color_cn_20230516.prototxt"
# options.china.seg_caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/seg_color_cn_20230516/seg_color_cn_20230516.caffemodel"
# seg_color_cn_20230530
options.china.seg_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/seg_color_cn_20230530/seg_color_cn_20230530.prototxt"
options.china.seg_caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/seg_color_cn_20230530/seg_color_cn_20230530.caffemodel"
options.china.seg_city_dict_name = None
options.china.seg_color_dict_name = "script.lpr.dataset.dataset_cn.dataset_dict.dataset_cn_dict_color"
options.china.seg_input_shape = (128, 64)
options.china.seg_city_bool = False
options.china.seg_color_bool = True

# 车牌长宽阈值
options.plate_height = [30, 250]  # zg chn
options.plate_width = [65, 500]

options.lpr_ocr_width_expand_ratio = 0.08
options.lpr_ocr_column_threshold = 2.5


###########################################
# sort
###########################################
options.sort_type = "car"

# car
options.max_age = 10
options.min_hits = 3 
options.iou_threshold = 0.3


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

options.capture_lpr_score_threshold = 0.8
options.capture_lpr_num_frame_threshold = 4
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