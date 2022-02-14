import argparse
import cv2
import numpy as np
import os
import sys 
import torch
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
from regreesion2d.plate_regreesion.infer.plate_regression import PlateRegression
from regreesion2d.plate_regreesion.utils.draw_tools import draw_detection_result
from Basic.script.xml.xml_add import XmlAdder


def car_bboxes_filter(bbox, mode='xywh', area_threshold = 4000):
    if not len(bbox):
        return []

    bbox = np.array(bbox)

    if mode == 'xywh':
        area = bbox[:, 2] * bbox[:, 3]
    elif mode == 'ltrb':
        area = ((bbox[:, 2]-bbox[:, 0]) *
              (bbox[:, 3]-bbox[:, 1]))
    else:
        print("Unknown mode")
        return None
    
    print(area, area >= area_threshold)
    bbox = bbox[area >= area_threshold]

    return bbox


def plate_bboxes_filter(bbox, mode='xywh', area_threshold = 100):
    if not len(bbox):
        return []

    bbox = np.array(bbox)

    if mode == 'xywh':
        area = bbox[:, 2] * bbox[:, 3]
    elif mode == 'ltrb':
        area = ((bbox[:, 2]-bbox[:, 0]) *
              (bbox[:, 3]-bbox[:, 1]))
    else:
        print("Unknown mode")
        return None
    
    # print(area, area >= area_threshold)
    bbox = bbox[area >= area_threshold]

    return bbox
    

def inference_images(args):
    # mkdir 
    if not os.path.isdir(args.output_img_dir):
        os.makedirs(args.output_img_dir)

    if not os.path.isdir(args.output_xml_dir):
        os.makedirs(args.output_xml_dir)

    # model init
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    plate_detector = PlateRegression(args.plate_model_path, args.plate_config_path, device)

    # image init 
    img_list = np.array(os.listdir(args.img_dir))
    img_list = img_list[[jpg.endswith('.jpg') for jpg in img_list]]
    img_list = img_list[[os.path.exists(os.path.join(args.xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in img_list]]
    img_list.sort()
    
    for idx in tqdm(range(len(img_list))):
        img_path = os.path.join(args.img_dir, img_list[idx])
        xml_path = os.path.join(args.xml_dir, img_list[idx].replace(".jpg", ".xml"))
        output_img_path = os.path.join(args.output_img_dir, img_list[idx])
        output_xml_path = os.path.join(args.output_xml_dir, img_list[idx].replace(".jpg", ".xml"))
        
        tqdm.write(img_path)

        img = cv2.imread(img_path)
        bboxes = {}
        bboxes['car'] = []

        # load car bboxes
        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

        for obj in root.iter('object'):
            classname = obj.find('name').text.lower().strip()
            
            if classname in args.select_name_list:
                bbox_obj = obj.find('bndbox')

                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox_obj = []
                for i, pt in enumerate(pts):
                    cur_pt = int(float(bbox_obj.find(pt).text)) - 1
                    bndbox_obj.append(cur_pt)
                
                bboxes['car'].append(bndbox_obj)

        # bboxes filter
        bboxes['car'] = car_bboxes_filter(bboxes['car'], mode='ltrb')

        # plate_detector
        bboxes["plate"] = plate_detector.detect(img, bboxes['car'])

        # bboxes filter
        bboxes["plate"] = plate_bboxes_filter(bboxes["plate"], mode='ltrb')

        # draw img
        img = draw_detection_result(img, bboxes, mode='ltrb')
        cv2.imwrite(output_img_path, img)

        # xml add
        xml_adder = XmlAdder(xml_path)
        xml_adder.add_object("plate", bboxes["plate"])
        xml_adder.write_xml(output_xml_path)


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.plate_model_path = "/mnt/huanyuan2/model/image_model/license_plate_model_wjh/国内、新加坡/MobileNetSmallV1_PO_singapore_Wdm_2020_03_19_17_36_30_PT_best/MobileNetSmallV1_PO_singapore_Wdm_2020_03_19_17_36_30_PT_best.pt"
    args.plate_config_path = "/home/huanyuan/code/demo/Image/regreesion2d/plate_regreesion/config/config.py"
    args.img_dir = "/mnt/huanyuan2/data/image/RM_ADAS_AllInOne/allinone/JPEGImages"
    args.xml_dir = "/mnt/huanyuan2/data/image/RM_ADAS_AllInOne/allinone/XML"
    args.output_img_dir = "/mnt/huanyuan2/data/image/RM_ADAS_AllInOne/allinone/test"
    args.output_xml_dir = "/mnt/huanyuan2/data/image/RM_ADAS_AllInOne/allinone/test_xml"

    args.select_name_list = ["car_front", "car_reg", "car_big_front", "car_big_reg"]
    inference_images(args)


if __name__ == '__main__':
    main()
