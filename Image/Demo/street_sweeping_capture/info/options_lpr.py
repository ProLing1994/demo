from easydict import EasyDict as edict

lpr = edict()


###########################################
# detector
###########################################

# ssd
# 2022-07-22-00
# pytorch 
# lpr.ssd_prototxt = None
# lpr.ssd_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-07-22-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
# caffe
# lpr.ssd_prototxt = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-07-22-00/FPN_RFB_3class_3attri_noDilation_prior.prototxt"
# lpr.ssd_model_path = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-07-22-00/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-07-22-00.caffemodel"
# openvino
# lpr.ssd_prototxt = None
# lpr.ssd_model_path = "/mnt/huanyuan/model_final/image_model/zg/gvd_ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-07-22-00/openvino_model/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-07-22-00.xml"

# 2022-08-10-00
# pytorch 
# lpr.ssd_prototxt = None
# lpr.ssd_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-08-10-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_zf_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
# caffe
# lpr.ssd_prototxt = ""
# lpr.ssd_model_path = ""
# openvino
# lpr.ssd_prototxt = None
# lpr.ssd_model_path = ""

## car_bus_truck_motorcyclist_licenseplate_motolicenseplate_softmax
# pytorch 
# lpr.ssd_prototxt = None
# lpr.ssd_model_path = ""
# caffe
lpr.ssd_prototxt = "/mnt/huanyuan/model_final/image_model/schoolbus/ssd_rfb/car_bus_truck_motorcyclist_licenseplate_motolicenseplate_softmax/FPN_RFB_4class_3attri_noDilation_prior.prototxt"
lpr.ssd_model_path = "/mnt/huanyuan/model_final/image_model/schoolbus/ssd_rfb/car_bus_truck_motorcyclist_licenseplate_motolicenseplate_softmax/SSD_VGG_FPN_RFB_VOC_car_bus_truck_motorcyclist_licenseplate_motolicenseplate_2023_03_06.caffemodel"
# openvino
# lpr.ssd_prototxt = None
# lpr.ssd_model_path = ""

# yolov6
lpr.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_jpf/yolov6.py"
lpr.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_jpf/epoch_260.pth"
lpr.yolov6_class_name = ['car', 'bus', 'truck', 'car_reg', 'car_big_reg', 'car_front',
                        'car_big_front', 'person', 'motorcyclist', 'bicyclist',
                        'sign_upspeed_round', 'sign_upspeed_square', 'sign_stop', 'sign_height',
                        'light_share0', 'light_share', 'bridge', 'zebra_crossing', 'license_plate']
lpr.yolov6_threshold_list = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

# 是否将 car\bus\truck 合并为一类输出
# lpr.car_attri_merge_bool = True
lpr.car_attri_merge_bool = False
lpr.car_attri_merge_name = 'car_bus_truck'
lpr.car_attri_name_list = [ 'car', 'bus', 'truck', 'motorcyclist' ]
lpr.license_plate_name = 'license_plate'


###########################################
# lpr
###########################################
lpr.lpr_caffe_bool = True
lpr.lpr_pytorch_bool = False

################
# china
################
lpr.china = edict()
# 0628
# lpr.china.ocr_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/0628/china_double_softmax.prototxt"
# lpr.china.ocr_caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/0628/china_double.caffemodel"
# lpr.china.input_shape = (256, 64)
# lpr.china.ocr_labels = ["-","皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
#                             "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
#                             "陕", "甘", "青", "宁", "新", "警", "学", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
#                             'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
#                             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '挂']
# lpr.china.ocr_prefix_beam_search_bool = False

# ocr_cn_20230512
lpr.china.ocr_pth_path = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/ocr_cn_20230512/crnn_best.pth"
lpr.china.ocr_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/ocr_cn_20230512/cnn_256x64_73.prototxt"
lpr.china.ocr_caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/ocr_cn_20230512/ocr_cn_20230512.caffemodel"
lpr.china.input_shape = (256, 64)
lpr.china.ocr_labels = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
                            "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
                            "陕", "甘", "青", "宁", "新", "警", "学", '挂', '领', '空', '港', '澳', 
                            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 
                            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']
lpr.china.ocr_prefix_beam_search_bool = False

# color
# seg_color_cn_20230512
lpr.china.seg_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/seg_color_cn_20230512/seg_color_cn_20230512.prototxt"
lpr.china.seg_caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr/lpr_zg/china/seg_color_cn_20230512/seg_color_cn_20230512.caffemodel"
lpr.china.seg_city_dict_name = None
lpr.china.seg_color_dict_name = "script.lpr.dataset.dataset_cn.dataset_dict.dataset_cn_dict_color"
lpr.china.seg_input_shape = (128, 64)
lpr.china.seg_city_bool = False
lpr.china.seg_color_bool = True

# 车牌长宽阈值
lpr.plate_height = [20, 250]
lpr.plate_width = [65, 500]

lpr.lpr_ocr_width_expand_ratio = 0.05
lpr.lpr_ocr_column_threshold = 2.5


###########################################
# sort
###########################################
lpr.sort_type = "car"

# car
lpr.max_age = 10
lpr.min_hits = 3 
lpr.iou_threshold = 0.3
