import argparse
import cv2
import json
import os
import sys 
import shutil
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
# sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.xml.xml_write import write_xml
from Basic.utils.folder_tools import *

ignore_dataset = [ ]


def cal_iou(bbox1, bbox2):
    ixmin = max(bbox1[0], bbox2[0])
    iymin = max(bbox1[1], bbox2[1])
    ixmax = min(bbox1[2], bbox2[2])
    iymax = min(bbox1[3], bbox2[3])
    iw = max(ixmax - ixmin, 0.)
    ih = max(iymax - iymin, 0.)
    inters = iw * ih
    uni = ((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) + \
            (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) - \
            inters)
    overlaps = inters / uni
    return overlaps


def label_to_xml(img_folder, gt_path, output_folder, args):
    img_list = os.listdir(img_folder)
    img_list.sort()

    for idx in tqdm(range(len(img_list))):
        
        img_name = img_list[idx]

        # json
        gt_dict = {}
        json_path = os.path.join(gt_path, img_name.replace('_leftImg8bit.png', '_gtFine_polygons.json'))
        with open(json_path, 'r', encoding='UTF-8') as fr:
            try:
                annotation = json.load(fr)
            except:
                raise Exception()  

        img_shape = [annotation['imgWidth'], annotation['imgHeight'], 3]

        for object_idx in annotation['objects']:
            label = object_idx['label']
            if label in args.select_name_list:
                x1, y1, x2, y2 = img_shape[0], img_shape[1], 0, 0

                for point_idx in object_idx['polygon']:
                    x1 = min(x1, point_idx[0])
                    y1 = min(y1, point_idx[1])
                    x2 = max(x2, point_idx[0])
                    y2 = max(y2, point_idx[1])
                
                if label not in gt_dict:
                    gt_dict[label] = []
                gt_dict[label].append({'label': label, 'bbox':[x1, y1, x2, y2]})  

        # xml
        xml_bboxes = {}
        for label_idx in gt_dict.keys():

            if label_idx in ['car', 'bus', 'truck', 'trailer', 'train', 'caravan', 'license plate', 'person']:
                gt_idx = gt_dict[label_idx]

                for bbox_idx in range(len(gt_idx)):
                    gt_bbox_idx = gt_idx[bbox_idx]

                    if label_idx not in xml_bboxes:
                        xml_bboxes[label_idx] = []
                    xml_bboxes[label_idx].append([gt_bbox_idx['bbox'][0], gt_bbox_idx['bbox'][1], gt_bbox_idx['bbox'][2], gt_bbox_idx['bbox'][3]])
                
            elif label_idx in ['bicycle', 'bicyclegroup', 'motorcycle']:
                gt_idx = gt_dict[label_idx]

                if label_idx == 'bicyclegroup':
                    label_idx = 'bicycle'

                for bbox_idx in range(len(gt_idx)):
                    gt_bbox_idx = gt_idx[bbox_idx]
                    find_rider_bool = False
                    
                    if 'rider' in gt_dict.keys():
                        rider_idx = gt_dict['rider']

                        for bbox_idy in range(len(rider_idx)):
                            rider_bbox_idy = rider_idx[bbox_idy]

                            # 计算 iou，iou 大于 0.2 为对应骑行者
                            iou = cal_iou(gt_bbox_idx['bbox'], rider_bbox_idy['bbox'])
                            if iou > 0.1:
                                find_rider_bool = True
                                x1 = min(gt_bbox_idx['bbox'][0],  rider_bbox_idy['bbox'][0])
                                y1 = min(gt_bbox_idx['bbox'][1],  rider_bbox_idy['bbox'][1])
                                x2 = max(gt_bbox_idx['bbox'][2],  rider_bbox_idy['bbox'][2])
                                y2 = max(gt_bbox_idx['bbox'][3],  rider_bbox_idy['bbox'][3])
                                match_bbox = [x1, y1, x2, y2]

                                break
                    
                        if find_rider_bool == True:
                            if args.match_class_map[label_idx] not in xml_bboxes:
                                xml_bboxes[args.match_class_map[label_idx]] = []
                            xml_bboxes[args.match_class_map[label_idx]].append([match_bbox[0], match_bbox[1], match_bbox[2], match_bbox[3]]) 
                        else:
                            if label_idx not in xml_bboxes:
                                xml_bboxes[label_idx] = []
                            xml_bboxes[label_idx].append([gt_bbox_idx['bbox'][0], gt_bbox_idx['bbox'][1], gt_bbox_idx['bbox'][2], gt_bbox_idx['bbox'][3]]) 
                    else:
                        if label_idx not in xml_bboxes:
                            xml_bboxes[label_idx] = []
                        xml_bboxes[label_idx].append([gt_bbox_idx['bbox'][0], gt_bbox_idx['bbox'][1], gt_bbox_idx['bbox'][2], gt_bbox_idx['bbox'][3]])

        # check xml
        xml_bool = False
        for xml_key in xml_bboxes.keys():
            if xml_key in args.set_class_list:
                xml_bool = True

        if xml_bool:
            # img
            img_path = os.path.join(img_folder, img_name)
            output_path = os.path.join(output_folder, img_name.replace('.png', '.jpg'))
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            cv2.imwrite(output_path, img)

            xml_path = os.path.join(output_folder, img_name.replace('.png', '.xml'))
            write_xml(xml_path, img_path, xml_bboxes, img_shape)

def tranform(args):
    subfolder_list = os.listdir(args.input_dir)
    subfolder_list.sort()

    for subfolder_idx in tqdm(range(len(subfolder_list))):
        subfolder_name = subfolder_list[subfolder_idx]

        if subfolder_name in ignore_dataset:
            continue
        print(subfolder_name)

        img_folder = os.path.join( args.input_dir, subfolder_name ) 
        gt_folder = os.path.join( args.label_dir, subfolder_name ) 

        output_folder = os.path.join( args.output_dir, 'JPEGImages' ) 
        create_folder(output_folder)

        label_to_xml(img_folder, gt_folder, output_folder, args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.dataset_type = "train"       # 训练集
    # args.dataset_type = "val"           # 验证集

    args.input_dir = "/mnt/huanyuan/temp/Cityscapes/leftImg8bit/" + args.dataset_type + "/" 
    args.label_dir = "/mnt/huanyuan/temp/Cityscapes/gtFine/" + args.dataset_type + "/" 
    args.output_dir = "/mnt/huanyuan/temp/Cityscapes/xml_gt/" + args.dataset_type + "/" 

    args.select_name_list = ['car', 'bus', 'truck', 'trailer', 'train', 'caravan', 'license plate', 'person', 'bicycle', 'bicyclegroup', 'motorcycle', 'rider']
    args.set_class_list = ['car', 'bus', 'truck', 'trailer', 'train', 'caravan', 'license plate', 'person', 'bicycle', 'motorcycle', 'bicyclist', 'motorcyclist']
    args.match_class_map = {'bicycle': 'bicyclist', 'motorcycle': 'motorcyclist' }
    tranform(args)