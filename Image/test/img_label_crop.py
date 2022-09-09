# -*- coding: UTF-8 -*-
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import os
import random
import xml.etree.cElementTree as ET


def IOU(rec1, rec2):
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    # sum_area = S_rec1 + S_rec2
    sum_area = min(S_rec1,S_rec2)
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area)) * 1.0


def img_aug( image, target ):

    trans = iaa.Sequential([
        iaa.SomeOf((0, 2), [
            iaa.Sometimes(0.6, iaa.GaussianBlur(2.0)),
            iaa.Sometimes(0.6, iaa.ChangeColorTemperature((7000, 10000))),
            iaa.Sometimes(0.6, iaa.GammaContrast((0.5, 2.0))),
            iaa.Sometimes(0.6, iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255))),
            # iaa.Sometimes(0.6, iaa.Cutout(nb_iterations=(1, 5), size=(0.1, 0.2), squared=False)),
            iaa.Sometimes(0.6, iaa.Multiply((1.2, 1.5)))
        ]),
        iaa.Affine(shear=(-20, 20))
    ])

    bbox_list=[]
    for bbox in target:
        bbox_list.append(BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]))
    bbs = BoundingBoxesOnImage(bbox_list, shape=image.shape)

    image_aug, bbox_aug = trans(image=image, bounding_boxes=bbs)

    target_aug = target.copy()
    for idx in range(len(target)):
        target_aug[idx] = [bbox_aug[idx].x1_int, bbox_aug[idx].y1_int, bbox_aug[idx].x2_int, bbox_aug[idx].y2_int, target[idx][-1]]

    return image_aug, target_aug


def find_plate_bbox(target_aug):

    # init
    car_list = []
    plate_list = []
    plate_bbox_find = []

    for idx in range(len(target_aug)):
        if('plate' in VOC_CLASSES[target_aug[idx][-1]]):
            plate_list.append({'target':target_aug[idx], 'index':idx, 'match_flag':False, 'match_index':-1})
        else:
            car_list.append({'target':target_aug[idx], 'index':idx, 'match_flag':False})

    # init
    max_area = 0
    max_plate_id = -1

    # 寻找有车牌且面积最大的车辆作为目标
    for plate_obj in plate_list:
        for car_obj in car_list:
            
            if(car_obj['match_flag']):
                continue

            plate_bbox = plate_obj['target']
            car_bbox = car_obj['target']
            half_car = car_bbox.copy()
            half_car[1] = half_car[1] + int(half_car[3] * 0.5 - half_car[1] * 0.5)
            iou = IOU(plate_bbox, half_car)

            if(iou > 0.95):
                car_obj['match_flag'] = True
                car_obj['match_index'] = plate_obj['index']
                car_obj['car_area'] = (car_bbox[2] - car_bbox[0] ) * ( car_bbox[3] - car_bbox[1] )
                break

    for car_obj in car_list:
        if(car_obj['match_flag']):
            if(car_obj['car_area'] > max_area):
                max_area = car_obj['car_area']
                max_plate_id = car_obj['match_index']
    
    if max_plate_id != -1:

        plate_bbox_find = target_aug[max_plate_id]
    
    return plate_bbox_find
    

def crop_img_bbox(image, target, bbox_find):

    expand_ratio = random.randint(6, 12)
    left_dist = random.randint(20, 70) / 100
    top_dist = random.randint(20, 50) / 100

    # crop
    roi_w = (bbox_find[2] - bbox_find[0]) * expand_ratio
    roi_h = roi_w * 3 / 4
    min_x = int(max(0, (bbox_find[2] + bbox_find[0]) / 2 - left_dist * roi_w))
    min_y = int(max(0, (bbox_find[1] + bbox_find[3]) / 2 - top_dist * roi_h))
    max_x = int(min(image.shape[1], min_x + roi_w))
    max_y = int(min(image.shape[0], min_y + roi_h))

    # crop img
    image = image[min_y: max_y, min_x: max_x]
    
    # clear bbox 
    target = clear_bbox(target, [min_x, min_y, max_x, max_y])

    # crop bbox 
    target = crop_bbox(target, [min_x, min_y, max_x, max_y])

    return image, target


def resize_img_bbox(image, target):

    height, width, _ = image.shape
    input_size=(width, height)
    scale_x = width / input_size[0]
    scale_y = height / input_size[1]

    # resize img 
    image = cv2.resize(image, input_size)

    # resize bbox 
    target = resize_bbox(target, scale_x, scale_y)

    return image, target


def clear_bbox(target, roi):
    id = 0
    while id < len( target ):
        if( IOU(target[id], roi) > 0.1 ):   # 剔除了那些过于残破的目标
            id += 1
        else:
            target.pop( id )

    return target


def crop_bbox(target, roi):
    
    for id in range(len(target)):
        target[id][0] = max(0, target[id][0] - roi[0])
        target[id][1] = max(0, target[id][1] - roi[1])
        target[id][2] = min(roi[2], target[id][2]) - roi[0]
        target[id][3] = min(roi[3], target[id][3]) - roi[1]
            
    return target


def resize_bbox(target, scale_x, scale_y):
    
    for id in range(len(target)):
        target[id][0] = int(target[id][0] / scale_x)
        target[id][1] = int(target[id][1] / scale_y)
        target[id][2] = int(target[id][2] / scale_x)
        target[id][3] = int(target[id][3] / scale_y)
            
    return target


def preproc( image, target ):
    
    # aug
    image_aug, target_aug = img_aug( image, target )

    # find plate bbox
    plate_bbox_find = find_plate_bbox(target_aug)


    if len(plate_bbox_find):

        # crop_img_bbox
        image, target = crop_img_bbox(image_aug, target_aug, plate_bbox_find)
        image, target = resize_img_bbox(image, target)

    else:
        # resize
        image, target = resize_img_bbox(image, target)

    return image, target


if __name__=='__main__':
    data_root = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/test/"
    file_list = os.listdir(data_root)
    VOC_CLASSES = ('__background__', 'background_all', 'license_plate', 'car', 'bus', 'truck', 'background_carbustruck',
                   'background_licenseplate', 'car_bus_truck', "roi_ignore_plate", 'license_plate_ignore', 'neg')

    for f in range(len(file_list)):
        file_name = os.path.basename(file_list[f])
        if not file_name.endswith(".jpg"):
            continue

        image_path = os.path.join(data_root, file_name)
        xml_path = os.path.join(data_root, file_name.replace('.jpg', '.xml'))
        img = cv2.imread(image_path)
        target_root = ET.parse(xml_path).getroot()
        target = []
        for child in target_root:
            if (child.tag == 'size'):
                width = child[0].text
                height = child[1].text
                depth = child[2].text
            if (child.tag == 'object'):
                for child_2nd in child:
                    if (child_2nd.tag == 'name'):
                        obj_name = child_2nd.text
                    if (child_2nd.tag == 'bndbox'):
                        xmin = int(child_2nd[0].text)
                        ymin = int(child_2nd[1].text)
                        xmax = int(child_2nd[2].text)
                        ymax = int(child_2nd[3].text)
                if('plate' in obj_name):
                    obj_name='license_plate'
                target.append([xmin, ymin, xmax, ymax, VOC_CLASSES.index(obj_name)])

        img, target = preproc( img, target ) # 预处理函数
        # for sample in target:
        #     cv2.rectangle(img, (sample[0], sample[1]), (sample[2], sample[3]), [0, 255, 0], 1, 1)
        #     cv2.putText(img, VOC_CLASSES[sample[-1]], (sample[0], sample[1] + 20), 1, 1, [0, 255, 255], 1)

        output_path = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/test_res/" + file_name
        cv2.imwrite(output_path, img)
        # cv2.imshow('a', img)
        # cv2.waitKey(-1)
