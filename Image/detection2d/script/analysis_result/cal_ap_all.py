import argparse
import numpy as np
import os
import sys

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.script.analysis_result.cal_ap_merge import voc_eval, voc_ap


def calculate_ap_all(args):
    # init
    tp_dict = {}
    fp_dict = {}
    conf_dict = {}
    npos_dict = {}

    for idx in range(len(args.data_list)):
        data_name_idx = args.data_list[idx]
        print('Date name = {}'.format(data_name_idx))
        
        if args.from_dataset_bool:
            args.imageset_file = os.path.join(args.data_dir, data_name_idx, "ImageSets/Main/" + test_name + ".txt")
            args.anno_dir = os.path.join(args.data_dir, data_name_idx, args.anno_name)                 
            args.jpg_dir = os.path.join(args.data_dir, data_name_idx,  "JPEGImages/")

            args.det_path_dict = { 'car': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')) + '_' + test_name, 'results/det_test_car.txt'),
                                'bus': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')) + '_' + test_name, 'results/det_test_bus.txt'),
                                'truck': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')) + '_' + test_name, 'results/det_test_truck.txt'),
                                'license_plate': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')) + '_' + test_name, 'results/det_test_license_plate.txt'),
                                'car_bus_truck': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')) + '_' + test_name, 'results/det_test_car_bus_truck.txt'), 
                                'bus_truck': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')) + '_' + test_name, 'results/det_test_bus_truck.txt'), 
                                'non_motorized': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')) + '_' + test_name, 'results/det_test_non_motorized.txt'), 
                                'non_motorized_person': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')) + '_' + test_name, 'results/det_test_non_motorized_person.txt'), 
                                'person': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')) + '_' + test_name, 'results/det_test_person.txt'), 
                                } 

            anno_path = os.path.join(args.anno_dir, '%s.xml')
            cache_dir = os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')) + '_' + test_name, 'results')

        else:
            args.imageset_file = os.path.join(args.data_dir, data_name_idx, "images.txt")
            args.anno_dir = os.path.join(args.data_dir, data_name_idx + '_' + args.anno_name)
            args.jpg_dir = os.path.join(args.data_dir, data_name_idx)

            args.det_path_dict = { 'car': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')), 'results/det_test_car.txt'),
                                'bus': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')), 'results/det_test_bus.txt'),
                                'truck': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')), 'results/det_test_truck.txt'),
                                'license_plate': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')), 'results/det_test_license_plate.txt'),
                                'car_bus_truck': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')), 'results/det_test_car_bus_truck.txt'), 
                                'bus_truck': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')), 'results/det_test_bus_truck.txt'), 
                                'non_motorized': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')), 'results/det_test_non_motorized.txt'), 
                                'non_motorized_person': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')), 'results/det_test_non_motorized_person.txt'), 
                                'person': os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')), 'results/det_test_person.txt'), 
                                } 

            anno_path = os.path.join(args.anno_dir, '%s.xml')
            cache_dir = os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')), 'results')

        # clear
        cache_file = os.path.join(cache_dir, 'annots.pkl')
        if os.path.exists(cache_file):
            os.remove(cache_file)

        for class_name in args.cal_ap_dict.keys():
            tp, fp, conf, npos, rec, prec, ap = voc_eval(
                merge_class_name=class_name,
                detpath_dict=args.det_path_dict, 
                annopath=anno_path, 
                imagesetfile=args.imageset_file, 
                classname=args.cal_ap_dict[class_name], 
                cachedir=cache_dir,
                ovthresh=args.over_thresh, 
                iou_uni_use_label_bool=args.iou_uni_use_label_bool,
                width_height_ovthresh_bool=args.width_height_over_thresh_bool,
                width_ovthresh=args.width_over_thresh,
                height_ovthresh=args.height_over_thresh,
                roi_set_bool=args.roi_set_bool,
                roi_set_bbox_2M=args.roi_set_bbox_2M,
                roi_set_bbox_5M=args.roi_set_bbox_5M,
                write_bool=args.write_bool,
                jpg_dir=args.jpg_dir,
                write_unmatched_bool=args.write_unmatched_bool,
                write_false_positive_bool=args.write_false_positive_bool,
                use_07_metric=args.use_07_metric)

            print('AP for {} = {:.3f} \n'.format(class_name, ap))

            if class_name in tp_dict:
                tp_dict[class_name] = np.concatenate((tp_dict[class_name], tp), axis=0)
                fp_dict[class_name] = np.concatenate((fp_dict[class_name], fp), axis=0)
                conf_dict[class_name] = np.concatenate((conf_dict[class_name], conf), axis=0)
                npos_dict[class_name] = npos_dict[class_name] + npos
            else:
                tp_dict[class_name] = tp
                fp_dict[class_name] = fp
                conf_dict[class_name] = conf
                npos_dict[class_name] = npos

    # cal ap
    for class_name in args.cal_ap_dict.keys():

        # sort by confidence
        sorted_ind = np.argsort(-conf_dict[class_name])
        tp = tp_dict[class_name][sorted_ind]
        fp = fp_dict[class_name][sorted_ind]
        conf = conf_dict[class_name][sorted_ind]
        npos = npos_dict[class_name]

        # compute precision recall
        fp_sum = np.cumsum(fp)
        tp_sum = np.cumsum(tp)
        rec = tp_sum / float(npos)
        # print("npos: {}".format(npos))
        if npos != 0:
            print("recall(TPR): {:.3f}({}/{})".format(tp.sum()/ float(npos), tp.sum(), npos))
        else:
            print("tpr: None")
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp_sum / np.maximum(tp_sum + fp_sum, np.finfo(np.float64).eps)
        print("precision(PPV): {:.3f}({}/{})".format(tp.sum()/ (tp.sum() + fp.sum()), tp.sum(), (tp.sum() + fp.sum())))
        ap = voc_ap(rec, prec, args.use_07_metric)
        print('ALL AP for {} = {:.3f}\n'.format(class_name, ap))

    return 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ######################################
    # Car_Licenseplate
    ######################################
    # args.data_dir = "/yuanhuan/data/image/LicensePlate/China/"
    # args.imageset_file = args.data_dir + "ImageSets/Main/test.txt"
    # args.anno_dir =  args.data_dir + "Annotations_CarLicenseplate/"
    # args.jpg_dir =  os.path.join(args.data_dir,  "JPEGImages/")

    # # # ssd rfb
    # # args.det_path_dict = { 'car': '/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-02-11-16_focalloss_car_licenseplate/eval_epoches_299/LicensePlate_China_test/results/det_test_car.txt',
    # #                        'license_plate': '/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-02-11-16_focalloss_car_licenseplate/eval_epoches_299/LicensePlate_China_test/results/det_test_license_plate.txt',
    # #                      } 
    # # args.over_thresh = 0.4
    # # args.use_07_metric = False
    # # 是否保存识别结果和检出结果
    # args.write_bool = True
    # # 是否保存漏检结果
    # args.write_unmatched_bool = True
    # # args.output_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-02-11-16_focalloss_car_licenseplate/eval_epoches_299/LicensePlate_China_test/results/"

    # # yolox
    # args.det_path_dict = { 'car': '/yuanhuan/model/image/yolox_vgg/yoloxv2_vggrm_640_384_car_license_plate/eval_epoches_24/LicensePlate_China_xml/det_test_car.txt',
    #                        'license_plate': '/yuanhuan/model/image/yolox_vgg/yoloxv2_vggrm_640_384_car_license_plate/eval_epoches_24/LicensePlate_China_xml/det_test_license_plate.txt',
    #                      } 
    # # args.over_thresh = 0.35
    # args.over_thresh = 0.4
    # args.use_07_metric = False
    # # 是否保存识别结果和检出结果
    # args.write_bool = True
    # # 是否保存漏检结果
    # args.write_unmatched_bool = True
    # args.output_dir = "/yuanhuan/model/image/yolox_vgg/yoloxv2_vggrm_640_384_car_license_plate/eval_epoches_24/LicensePlate_China_xml/"

    # #####################################
    # # Car_Bus_Truck_Licenseplate
    # # 测试集图像
    # #####################################
    # args.data_dir = "/yuanhuan/data/image/"
    # # args.data_list = ['ZG_ZHJYZ_detection/jiayouzhan', 'ZG_ZHJYZ_detection/jiayouzhan_5M', 'ZG_ZHJYZ_detection/sandaofangxian', 'ZG_ZHJYZ_detection/anhuihuaibeigaosu', 'ZG_ZHJYZ_detection/shenzhentiaoqiao']
    # # args.data_list = ['ZG_ZHJYZ_detection/anhuihuaibeigaosu_night_diguangzhao']
    # # args.data_list = ['ZG_ZHJYZ_detection/jiayouzhan', 'ZG_ZHJYZ_detection/jiayouzhan_5M', 'ZG_ZHJYZ_detection/sandaofangxian', 'ZG_ZHJYZ_detection/anhuihuaibeigaosu', 'ZG_ZHJYZ_detection/shenzhentiaoqiao', 'ZG_ZHJYZ_detection/anhuihuaibeigaosu_night_diguangzhao']
    # # args.data_list = ['RM_ADAS_AllInOne/allinone_w_licenseplate']
    # args.data_list = ['ZG_ZHJYZ_detection/jiayouzhan', 'ZG_ZHJYZ_detection/jiayouzhan_5M', 'ZG_ZHJYZ_detection/sandaofangxian', 'ZG_ZHJYZ_detection/anhuihuaibeigaosu', 'ZG_ZHJYZ_detection/shenzhentiaoqiao', 'ZG_ZHJYZ_detection/anhuihuaibeigaosu_night_diguangzhao', 'RM_ADAS_AllInOne/allinone_w_licenseplate']

    # args.cal_ap_dict = { 'car': ['car'], 
    #                     'bus': ['bus'], 'truck': ['truck'], 
    #                     'bus_truck': ['bus', 'truck'], 
    #                     'car_bus_truck': ['car', 'bus', 'truck'], 
    #                     'license_plate': ['license_plate'] }
    # # args.cal_ap_dict = { 'license_plate': ['license_plate'] }

    # # SSD_VGG_FPN_RFB_2022-03-09-17_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-03-09-17_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/eval_epoches_299/"

    # # SSD_VGG_FPN_RFB_2022-04-25-18_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-04-25-18_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/eval_epoches_299/"

    # # # SSD_VGG_FPN_RFB_2022-05-27-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-05-27-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/eval_epoches_299/"

    # # SSD_VGG_FPN_RFB_2022-07-22-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-07-22-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/eval_epoches_299/"

    # # SSD_VGG_FPN_RFB_2022-07-29-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-07-29-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/eval_epoches_299/"

    # # SSD_VGG_FPN_RFB_2022-08-10-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_zf_w_fuzzy_plate
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-08-10-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_zf_w_fuzzy_plate/eval_epoches_299/"

    # # SSD_VGG_FPN_RFB_2022-09-09-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_zf_crop_w_fuzzy_plate
    # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-09-09-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_zf_crop_w_fuzzy_plate/eval_epoches_250/"

    # # yolov6
    # # args.model_dir = "/yuanhuan/model/image/yolov6/yolov6_zg_gvd_adas_zg_data_attribute_conv_0804/eval_epoches_300_0.4"
    # # args.model_dir = "/yuanhuan/model/image/yolov6/yolov6_zg_gvd_adas_zg_data_attribute_conv_0809/eval_epoches_300_0.4"

    # # args.anno_name = 'Annotations_CarBusTruckLicenseplate_w_height'             # 高度大于 24 的 清晰车牌
    # # args.anno_name = 'Annotations_CarBusTruckLicenseplate_w_fuzzy_w_height'     # 高度大于 24 的 清晰车牌 & 模糊车牌
    # # args.anno_name = 'Annotations_CarBusTruckLicenseplate'                      # 清晰车牌
    # args.anno_name = 'Annotations_CarBusTruckLicenseplate_w_fuzzy'              # 清晰车牌 & 模糊车牌

    # args.from_dataset_bool = True

    ######################################
    # Car_Bus_Truck_Licenseplate
    # 收集测试图像：
    ######################################

    args.data_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/"

    # args.data_list = ['jiayouzhan_test_image/2MB']
    # args.data_list = ['jiayouzhan_test_image/2MB', 'jiayouzhan_test_image/2MH' ]
    # args.data_list = ['jiayouzhan_test_image/5MB', 'jiayouzhan_test_image/5MH' ]
    # args.data_list = ['jiayouzhan_test_image/SZTQ' ]
    # args.data_list = ['jiayouzhan_test_image/SDFX_B1', 'jiayouzhan_test_image/SDFX_B2', 'jiayouzhan_test_image/SDFX_H1', 'jiayouzhan_test_image/SDFX_H2' ]
    # args.data_list = ['jiayouzhan_test_image/TXSDFX_6', 'jiayouzhan_test_image/TXSDFX_7', 'jiayouzhan_test_image/TXSDFX_9', 'jiayouzhan_test_image/TXSDFX_c' ]
    # args.data_list = ['jiayouzhan_test_image/AHHBAS_41a', 'jiayouzhan_test_image/AHHBAS_41c', 'jiayouzhan_test_image/AHHBAS_43c', 'jiayouzhan_test_image/AHHBAS_418' ]
    # args.data_list = ['jiayouzhan_test_image/AHHBAS_kakou1', 'jiayouzhan_test_image/AHHBAS_kakou2', 'jiayouzhan_test_image/AHHBAS_kakou3', 'jiayouzhan_test_image/AHHBAS_kakou4' ]
    # args.data_list = ['jiayouzhan_test_image/AHHBPS' ]
    # args.data_list = ['jiayouzhan_test_image/2MB', 'jiayouzhan_test_image/2MH', 'jiayouzhan_test_image/5MB', 'jiayouzhan_test_image/5MH',
    #                   'jiayouzhan_test_image/SZTQ', 
    #                   'jiayouzhan_test_image/SDFX_B1', 'jiayouzhan_test_image/SDFX_B2', 'jiayouzhan_test_image/SDFX_H1', 'jiayouzhan_test_image/SDFX_H2',
    #                   'jiayouzhan_test_image/TXSDFX_6', 'jiayouzhan_test_image/TXSDFX_7', 'jiayouzhan_test_image/TXSDFX_9', 'jiayouzhan_test_image/TXSDFX_c', 
    #                   'jiayouzhan_test_image/AHHBAS_41a', 'jiayouzhan_test_image/AHHBAS_41c', 'jiayouzhan_test_image/AHHBAS_43c', 'jiayouzhan_test_image/AHHBAS_418', 
    #                   'jiayouzhan_test_image/AHHBAS_kakou1', 'jiayouzhan_test_image/AHHBAS_kakou2', 'jiayouzhan_test_image/AHHBAS_kakou3', 'jiayouzhan_test_image/AHHBAS_kakou4',                      
    #                   'jiayouzhan_test_image/AHHBPS' ]
    # args.data_list = ['jiayouzhan_test_image/AHHBAS_kakou2_night', 'jiayouzhan_test_image/AHHBAS_kakou3_night']
    args.data_list = ['jiayouzhan_test_image/2MB', 'jiayouzhan_test_image/2MH', 'jiayouzhan_test_image/5MB', 'jiayouzhan_test_image/5MH',
                      'jiayouzhan_test_image/SZTQ', 
                      'jiayouzhan_test_image/SDFX_B1', 'jiayouzhan_test_image/SDFX_B2', 'jiayouzhan_test_image/SDFX_H1', 'jiayouzhan_test_image/SDFX_H2',
                      'jiayouzhan_test_image/TXSDFX_6', 'jiayouzhan_test_image/TXSDFX_7', 'jiayouzhan_test_image/TXSDFX_9', 'jiayouzhan_test_image/TXSDFX_c', 
                      'jiayouzhan_test_image/AHHBAS_41a', 'jiayouzhan_test_image/AHHBAS_41c', 'jiayouzhan_test_image/AHHBAS_43c', 'jiayouzhan_test_image/AHHBAS_418', 
                      'jiayouzhan_test_image/AHHBAS_kakou1', 'jiayouzhan_test_image/AHHBAS_kakou2', 'jiayouzhan_test_image/AHHBAS_kakou3', 'jiayouzhan_test_image/AHHBAS_kakou4',                      
                      'jiayouzhan_test_image/AHHBPS',
                      'jiayouzhan_test_image/AHHBAS_kakou2_night', 'jiayouzhan_test_image/AHHBAS_kakou3_night' ]

    args.cal_ap_dict = { 'car': ['car'], 
                        'bus': ['bus'], 'truck': ['truck'], 
                        'bus_truck': ['bus', 'truck'], 
                        'car_bus_truck': ['car', 'bus', 'truck'], 
                        'license_plate': ['license_plate'] }
    # args.cal_ap_dict = { 'license_plate': ['license_plate'] }

    # SSD_VGG_FPN_RFB_2022-03-09-17_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate
    # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-03-09-17_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/eval_epoches_299/"

    # # SSD_VGG_FPN_RFB_2022-04-25-18_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate
    # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-04-25-18_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/eval_epoches_299/"

    # SSD_VGG_FPN_RFB_2022-05-27-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate
    # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-05-27-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/eval_epoches_299/"
    
    # # SSD_VGG_FPN_RFB_2022-07-22-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate
    # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-07-22-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/eval_epoches_299/"

    # SSD_VGG_FPN_RFB_2022-08-10-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_zf_w_fuzzy_plate
    # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-08-10-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_zf_w_fuzzy_plate/eval_epoches_299/"

    # SSD_VGG_FPN_RFB_2022-09-09-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_zf_crop_w_fuzzy_plate
    args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-09-09-00_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_zf_crop_w_fuzzy_plate/eval_epoches_250/"

    # yolov6
    # args.model_dir = "/yuanhuan/model/image/yolov6/yolov6_zg_gvd_adas_zg_data_attribute_conv_0804/eval_epoches_300_0.4"
    # args.model_dir = "/yuanhuan/model/image/yolov6/yolov6_zg_gvd_adas_zg_data_attribute_conv_0809/eval_epoches_300_0.4"

    # args.anno_name = 'Annotations_CarBusTruckLicenseplate_w_height'             # 高度大于 24 的 清晰车牌
    # args.anno_name = 'Annotations_CarBusTruckLicenseplate_w_fuzzy_w_height'     # 高度大于 24 的 清晰车牌 & 模糊车牌
    # args.anno_name = 'Annotations_CarBusTruckLicenseplate'                      # 清晰车牌
    args.anno_name = 'Annotations_CarBusTruckLicenseplate_w_fuzzy'              # 清晰车牌 & 模糊车牌

    args.from_dataset_bool = False

    # ######################################
    # # Nonmotorized_Car_Person
    # ######################################
    # args.data_dir = "/yuanhuan/data/image/"

    # # yolox
    # # args.data_list = ['ZG_BMX_detection/daminghu']
    # # args.data_list = ['ZG_BMX_detection/daminghu_night']
    # # args.data_list = ['ZG_BMX_detection/shandongyingzikou']
    # # args.data_list = ['ZG_BMX_detection/shandongyingzikou_night_hongwai']
    # # args.data_list = ['ZG_BMX_detection/yongzou_night_hongwai']
    # # args.data_list = ['ZG_BMX_detection/shenzhenlukou']
    # # args.data_list = ['ZG_BMX_detection/shenzhenlukou_night_hongwai']
    # # args.data_list = ['ZG_BMX_detection/shenzhenlukou_night_diguangzhao']
    # # args.data_list = ['ZG_BMX_detection/rongheng']
    # # args.data_list = ['ZG_BMX_detection/rongheng_night_hongwai']
    # args.data_list = ['ZG_BMX_detection/daminghu', 
    #                     'ZG_BMX_detection/daminghu_night', 
    #                     'ZG_BMX_detection/shandongyingzikou', 
    #                     'ZG_BMX_detection/shandongyingzikou_night_hongwai',
    #                     'ZG_BMX_detection/yongzou_night_hongwai', 
    #                     'ZG_BMX_detection/shenzhenlukou', 
    #                     'ZG_BMX_detection/shenzhenlukou_night_hongwai', 
    #                     'ZG_BMX_detection/shenzhenlukou_night_diguangzhao',
    #                     'ZG_BMX_detection/rongheng',
    #                     'ZG_BMX_detection/rongheng_night_hongwai']
    # # args.data_list = ['ZG_BMX_detection/rongheng',
    # #                     'ZG_BMX_detection/rongheng_night_hongwai']
    # # args.data_list = ['ZG_BMX_detection/daminghu', 
    # #                     'ZG_BMX_detection/daminghu_night']
                        
    # # args.cal_ap_dict = { 'car': ['car'], 
    # #                     'bus': ['bus'], 
    # #                     'truck': ['truck'], 
    # #                     'bus_truck': ['bus', 'truck'], 
    # #                     'car_bus_truck': ['car', 'bus', 'truck'], 
    # #                     'non_motorized': ['bicyclist', 'motorcyclist'], 
    # #                     'person': ['person'] }

    # args.cal_ap_dict = {'car_bus_truck': ['car', 'bus', 'truck'], 
    #                     'non_motorized': ['bicyclist', 'motorcyclist'], 
    #                     'person': ['person']}

    # # args.cal_ap_dict = {'person': ['person']}

    # # yolox
    # # args.model_dir = "/yuanhuan/model/image/yolox_vgg/car_non_motorized_person_yolox_model_zph/eval_epoches_24/"

    # # SSD_VGG_FPN_RFB_2022-06-20-12_focalloss_3class_car_bus_truck_non_motorized_person_ZG
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-06-20-12_focalloss_3class_car_bus_truck_non_motorized_person_ZG/eval_epoches_299/"
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-06-27-21_focalloss_3class_car_bus_truck_non_motorized_person_ZG/eval_epoches_299/"
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-06-27-21_focalloss_3class_car_bus_truck_non_motorized_person_ZG/eval_epoches_299/"
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-06-29-15_focalloss_3class_car_bus_truck_non_motorized_person_ZG/eval_epoches_299_0.3/"
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-06-29-15_focalloss_3class_car_bus_truck_non_motorized_person_ZG/eval_epoches_299_0.3_nms/"
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-06-30-19_focalloss_3class_car_bus_truck_non_motorized_person_ZG_eqlv2/eval_epoches_299/"
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-06-30-19_focalloss_3class_car_bus_truck_non_motorized_person_ZG_eqlv2/eval_epoches_499_0.5_nms/"
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-06-30-19_focalloss_3class_car_bus_truck_non_motorized_person_ZG_eqlv2/eval_epoches_499_0.7_nms/"

    # # SSD_VGG_FPN_RFB_2022-07-04-21_focalloss_3class_car_bus_truck_non_motorized_person_softmax_ADAS_BAD_ZG
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-07-04-21_focalloss_3class_car_bus_truck_non_motorized_person_softmax_ADAS_BAD_ZG/eval_epoches_299_0.3/"
    
    # # # SSD_VGG_FPN_RFB_2022-06-30-15_focalloss_1class_person_ADAS_BAD_ZG
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-06-30-15_focalloss_1class_person_ADAS_BAD_ZG/eval_epoches_299/"

    # # yolov6
    # # args.model_dir = "/yuanhuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_zg_data_0722/eval_epoches_300_0.4/"
    # args.model_dir = "/yuanhuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_zg_data_0824/eval_epoches_300_0.4/"
    # # args.model_dir = "/yuanhuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_zg_data_attribute_conv/eval_epoches_300_0.4/"

    # # args.anno_name = 'Annotations_CarBusTruckBicyclistMotorcyclistPerson'                             # 车 & 非机动 & 人 
    # args.anno_name = 'Annotations_CarBusTruckBicyclistMotorcyclistPerson_filter'                      # 车 & 非机动 & 人 & 大小过滤
    # # args.anno_name = 'Annotations_person_10_20'                                                       # 人(10_20)
    # # args.anno_name = 'Annotations_person_20_30'                                                       # 人(20_30)
    # # args.anno_name = 'Annotations_person_30_40'                                                       # 人(30_40)
    # # args.anno_name = 'Annotations_person_40_50'                                                       # 人(40_50)
    # # args.anno_name = 'Annotations_person_50_60'                                                       # 人(50_60)
    # # args.anno_name = 'Annotations_person_60_70'                                                       # 人(60_70)
    # # args.anno_name = 'Annotations_person_70_80'                                                       # 人(70_80)
    # # args.anno_name = 'Annotations_person_80_120'                                                      # 人(80_120)

    # args.from_dataset_bool = True

    # ######################################
    # # Nonmotorized_Car_Person
    # # 收集测试图像：
    # ######################################

    # args.data_dir = "/yuanhuan/data/image/ZG_BMX_detection/"

    # # args.data_list = ['banmaxian_test_image/2M_DaMingHu_far', 'banmaxian_test_image/2M_DaMingHu_near', 'banmaxian_test_image/2M_DaMingHu_night_far', 'banmaxian_test_image/2M_DaMingHu_night_near' ]
    # # args.data_list = ['banmaxian_test_image/2M_RongHeng_far', 'banmaxian_test_image/2M_RongHeng_near', 'banmaxian_test_image/2M_RongHeng_night_far', 'banmaxian_test_image/2M_RongHeng_night_near' ]
    # args.data_list = ['banmaxian_test_image/2M_DaMingHu_far', 'banmaxian_test_image/2M_DaMingHu_near', 'banmaxian_test_image/2M_DaMingHu_night_far', 'banmaxian_test_image/2M_DaMingHu_night_near',
    #                   'banmaxian_test_image/2M_RongHeng_far', 'banmaxian_test_image/2M_RongHeng_near', 'banmaxian_test_image/2M_RongHeng_night_far', 'banmaxian_test_image/2M_RongHeng_night_near', 
    #                   ]

    # args.cal_ap_dict = {'car_bus_truck': ['car', 'bus', 'truck'], 
    #                     'non_motorized': ['bicyclist', 'motorcyclist'], 
    #                     'person': ['person']}

    # # args.cal_ap_dict = {'person': ['person']}

    # # SSD_VGG_FPN_RFB_2022-06-29-15_focalloss_3class_car_bus_truck_non_motorized_person_ZG
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-06-29-15_focalloss_3class_car_bus_truck_non_motorized_person_ZG/eval_epoches_299/"

    # # SSD_VGG_FPN_RFB_2022-07-04-21_focalloss_3class_car_bus_truck_non_motorized_person_softmax_ADAS_BAD_ZG
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-07-04-21_focalloss_3class_car_bus_truck_non_motorized_person_softmax_ADAS_BAD_ZG/eval_epoches_299_0.3_nms/"
    
    # # # SSD_VGG_FPN_RFB_2022-06-30-15_focalloss_1class_person_ADAS_BAD_ZG
    # # args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-06-30-15_focalloss_1class_person_ADAS_BAD_ZG/eval_epoches_299/"

    # # yolov6
    # # args.model_dir = "/yuanhuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_zg_data_0722/eval_epoches_300_0.4/"
    # args.model_dir = "/yuanhuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_zg_data_0824/eval_epoches_300_0.4/"
    # # args.model_dir = "/yuanhuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_zg_data_attribute/eval_epoches_300_0.4/"

    # # args.anno_name = 'Annotations_CarBusTruckBicyclistMotorcyclistPerson'                             # 车 & 非机动 & 人 
    # args.anno_name = 'Annotations_CarBusTruckBicyclistMotorcyclistPerson_filter'                      # 车 & 非机动 & 人 & 大小过滤
    # # args.anno_name = 'Annotations_person_10_20'                                                       # 人(10_20)
    # # args.anno_name = 'Annotations_person_20_30'                                                       # 人(20_30)
    # # args.anno_name = 'Annotations_person_30_40'                                                       # 人(30_40)
    # # args.anno_name = 'Annotations_person_40_50'                                                       # 人(40_50)
    # # args.anno_name = 'Annotations_person_50_60'                                                       # 人(50_60)
    # # args.anno_name = 'Annotations_person_60_70'                                                       # 人(60_70)
    # # args.anno_name = 'Annotations_person_70_80'                                                       # 人(70_80)
    # # args.anno_name = 'Annotations_person_80_120'                                                      # 人(80_120)

    # args.from_dataset_bool = False

    #####################################
    # 分割线
    #####################################

    # test_name = "trainval"
    # test_name = "val"
    test_name = "test"

    args.over_thresh = 0.5
    args.use_07_metric = False

    # 是否设置 roi 区域，忽略边缘区域
    args.roi_set_bool = False
    # args.roi_set_bool = True
    args.roi_set_bbox_2M = [320, 360, 1600, 1080]   # 2M
    args.roi_set_bbox_5M = [432, 640, 2272, 1920]   # 5M

    # 是否在计算 iou 的过程中，计算 uni 并集的面积只关注 label 的面积
    args.iou_uni_use_label_bool = False

    # 是否关注车牌横向iou结果
    args.width_height_over_thresh_bool = False
    args.width_over_thresh = 0.9
    args.height_over_thresh = 0.75

    # 是否保存识别结果和检出结果
    args.write_bool = True

    # 是否保存漏检结果
    args.write_unmatched_bool = False

    # 是否保存假阳结果
    args.write_false_positive_bool = False

    calculate_ap_all(args)