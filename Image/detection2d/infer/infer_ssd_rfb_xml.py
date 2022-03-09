import argparse
import cv2
import numpy as np
import os
import sys 
import torch
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
from detection2d.ssd_rfb_crossdatatraining.test_tools import SSDDetector
from Basic.script.xml.xml_write import write_xml

from recognition2d.license_plate_recognition.infer.license_plate import license_palte_model_init_caffe, license_palte_crnn_recognition_caffe, license_palte_beamsearch_init, license_palte_crnn_recognition_beamsearch_caffe


def model_init(args):
    # model init
    merge_class_bool = False
    car_plate_detector = SSDDetector(model_path=args.ssd_car_plate_model_path, merge_class_bool=merge_class_bool)
    license_palte_detector = license_palte_model_init_caffe(args.plate_recognition_prototxt, args.plate_recognition_model_path)
    license_palte_beamsearch = license_palte_beamsearch_init()
    return (car_plate_detector, license_palte_detector, license_palte_beamsearch)


def img_detect(args, model, img):
    car_plate_detector = model[0]
    license_palte_detector = model[1]
    license_palte_beamsearch = model[2]

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # init
    show_bboxes = {}

    # car_plate_detector
    bboxes = car_plate_detector.detect(img)

    for key in bboxes.keys():
        if key in args.select_name_list:
            show_bboxes[key] = bboxes[key]

    # 区分清晰和模糊车牌
    for plate_idx in range(len(bboxes["license_plate"])):
        plate_bbox = bboxes["license_plate"][plate_idx]

        # crop
        crop_img = gray_img[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]

        # check
        if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
            continue

        # greedy ocr
        result_ocr, result_scors_list = license_palte_crnn_recognition_caffe(license_palte_detector, crop_img)

        # ocr 阈值判断
        if np.array(result_scors_list).mean() >= args.ocr_threshold:
            if args.plate_name in show_bboxes:
                show_bboxes[args.plate_name].append(plate_bbox)
            else:
                show_bboxes[args.plate_name] = [plate_bbox]
        else:
            if args.fuzzy_plate_name in show_bboxes:
                show_bboxes[args.fuzzy_plate_name].append(plate_bbox)
            else:
                show_bboxes[args.fuzzy_plate_name] = [plate_bbox]
            

    return show_bboxes


def inference_images(args):
    # mkdir 
    if args.write_bool:
        if not os.path.isdir(args.output_xml_dir):
            os.makedirs(args.output_xml_dir)

    # model init
    model = model_init(args)

    # image init 
    img_list = np.array(os.listdir(args.img_dir))
    img_list = img_list[[jpg.endswith('.jpg') for jpg in img_list]]
    img_list.sort()
    
    for idx in tqdm(range(len(img_list))):
        img_path = os.path.join(args.img_dir, img_list[idx])

        if args.write_bool:
            output_xml_path = os.path.join(args.output_xml_dir, img_list[idx].replace(".jpg", ".xml"))
        
        tqdm.write(img_path)

        img = cv2.imread(img_path)        

        # detect 
        show_bboxes = img_detect(args, model, img)

        # save xml
        if args.write_bool:
            write_xml(output_xml_path, img_path, show_bboxes, img.shape)


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.ssd_car_plate_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-02-24-15_focalloss_4class_car_bus_truck_licenseplate_zg_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
    args.plate_recognition_prototxt = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china_softmax.prototxt"
    args.plate_recognition_model_path = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china.caffemodel"

    # 是否保存结果
    args.write_bool = True

    # 是否设置 ocr 阈值挑选车牌
    args.ocr_threshold_bool = True
    args.ocr_threshold = 0.8
    
    # xml_name
    args.select_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate"]
    args.plate_name = "plate"
    args.fuzzy_plate_name = "fuzzy_plate"
    
    args.img_bool = True
    args.img_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试样本/sandaofangxian_test/"
    args.output_xml_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试样本_XML/sandaofangxian_test/"

    if args.img_bool:
        inference_images(args)


if __name__ == '__main__':
    main()