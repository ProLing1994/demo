import argparse
import cv2
import numpy as np
import os
import sys 
import torch
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.detection2d.ssd_rfb_crossdatatraining.test_tools import SSDDetector
from Image.Basic.script.xml.xml_write import write_xml


def img_detect(args, model, img):
    car_plate_detector = model[0]
    lpr = model[1]

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # init
    show_bboxes = {}

    # car_plate_detector
    bboxes = car_plate_detector.detect(img)

    for key in bboxes.keys():
        if key in args.select_name_list:
            show_bboxes[key] = bboxes[key]

    # 区分清晰和模糊车牌
    if "license_plate" in bboxes:
        for plate_idx in range(len(bboxes["license_plate"])):
            plate_bbox = bboxes["license_plate"][plate_idx]

            # crop
            crop_img = gray_img[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]

            # check
            if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                continue

            # greedy ocr
            result_ocr, result_scors_list = lpr.run(crop_img)

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


def pre_label(args):
    # mkdir 
    if not os.path.isdir(args.output_xml_dir):
        os.makedirs(args.output_xml_dir)

    # model init
    detector = SSDDetector(model_path=args.ssd_model_path, ssd_caffe_bool=False, ssd_openvino_bool=False, merge_class_bool=False, gpu_bool=True)

    # image init 
    img_list = np.array(os.listdir(args.input_dir))
    img_list = img_list[[jpg.endswith('.jpg') for jpg in img_list]]
    img_list.sort()
    
    for idx in tqdm(range(len(img_list))):

        img_name = img_list[idx]
        img_path = os.path.join(args.input_dir, img_name)
        output_xml_path = os.path.join(args.output_xml_dir, img_name.replace(".jpg", ".xml"))
        tqdm.write(img_path)

        img = cv2.imread(img_path)        

        # detector 
        bboxes = detector.detect( img, with_score=True )

        # select
        show_bboxes = {}
        for label_idx in range(len(args.pre_label_list)):
            if args.pre_label_list[label_idx] in bboxes:
                show_bboxes[args.pre_label_list[label_idx]] = bboxes[args.pre_label_list[label_idx]]

        # save xml
        write_xml(output_xml_path, img_path, show_bboxes, img.shape)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/mnt/huanyuan2/data/image/LicensePlate_ocr/SHATE/2023-02-09/"
    args.output_xml_dir = "/mnt/huanyuan2/data/image/LicensePlate_ocr/SHATE/xml-2023-02-09/"
    
    args.ssd_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2023-04-15-06_focalloss_6class_car_bus_truck_motorcyclist_licenseplate_motolicenseplate_softmax/SSD_VGG_FPN_RFB_VOC_epoches_268.pth"
    args.pre_label_list = ['license_plate']
    pre_label(args)
