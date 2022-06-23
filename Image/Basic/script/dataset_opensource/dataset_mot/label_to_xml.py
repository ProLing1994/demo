import argparse
import cv2
import os
import sys 
import shutil
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
# sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.xml.xml_write import write_xml
from Basic.utils.folder_tools import *

label_dict = {  "-1": 'none',
                "1": 'pedestrian',
                "2": 'person_on_vehicle', 
                "3": "car", 
                "4": "bicycle",
                "5": "motorbike",
                "6": "non_motorized_vehicle",
                "7": "static_person",
                "8": "distractor",
                "9": "occluder",
                "10": "occluder_on_the_ground",
                "11": "occluder_full",
                "12": "reflection",
                "13": "crowd",
            }

zg_label_dict = {   "1": 'person',
                    "3": "car_bus_truck",
                    "7": "person",
                }
                
ignore_label_id = ["2", "4", "5", "6", "8", "9", "10", "11", "12", "13"]
ignore_dataset = [ ]

def label_to_xml(img_folder, gt_path, output_folder, dataset_name, subfolder_name, args):

    # gt
    gt_dict = {}
    with open(gt_path, "r") as f:
        for line in f:
            line_str = line.strip().split(',')
            frame_num = line_str[0]
            id_num = line_str[1]
            bbox_ltwh = line_str[2: 6]

            if args.dataset_type == "train":
                if dataset_name == "MOT15":
                    class_id = "1"
                    visibility = "1"
                else:
                    class_id = line_str[7]
                    visibility = line_str[8]
            else:
                class_id = "1"
                visibility = "1"

            if frame_num not in gt_dict:
                gt_dict[frame_num] = []

            gt_dict[frame_num].append({'frame_num': frame_num, 'id_num':id_num, 'bbox_ltwh':bbox_ltwh, 'class_id':class_id, 'visibility':visibility})
    
    # label to xml
    for frame_idx in tqdm(gt_dict.keys()):
    
        # img
        img_name = "{:0>6d}.jpg".format(int(frame_idx))
        img_path = os.path.join(img_folder, img_name)
        output_path = os.path.join(output_folder, dataset_name + '_' + subfolder_name + '_' + img_name)
        shutil.copy(img_path, output_path)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_shape = [img.shape[1], img.shape[0], img.shape[2]]

        # xml
        xml_bboxes = {}
        gt_idx = gt_dict[frame_idx]
        for bbox_idx in range(len(gt_idx)):
            gt_bbox_idx = gt_idx[bbox_idx]

            # continue
            if gt_bbox_idx['class_id'] in ignore_label_id:
                continue
            if (gt_bbox_idx['class_id'] == "1" or gt_bbox_idx['class_id'] == "7") and float(gt_bbox_idx['visibility']) < args.person_visibility_threh:
                continue
            if (gt_bbox_idx['class_id'] == "3") and float(gt_bbox_idx['visibility']) < args.car_visibility_threh:
                continue

            # label = label_dict[gt_bbox_idx['class_id']]
            label = zg_label_dict[gt_bbox_idx['class_id']]
            x1 = int(float(gt_bbox_idx['bbox_ltwh'][0]))
            y1 = int(float(gt_bbox_idx['bbox_ltwh'][1]))
            x2 = int(float(gt_bbox_idx['bbox_ltwh'][0])) + int(float(gt_bbox_idx['bbox_ltwh'][2]))
            y2 = int(float(gt_bbox_idx['bbox_ltwh'][1])) + int(float(gt_bbox_idx['bbox_ltwh'][3]))

            x1 = max(0, x1)
            x1 = min(img_shape[0], x1)
            x2 = max(0, x2)
            x2 = min(img_shape[0], x2)
            y1 = max(0, y1)
            y1 = min(img_shape[1], y1)
            y2 = max(0, y2)
            y2 = min(img_shape[1], y2)

            if label not in xml_bboxes:
                xml_bboxes[label] = []

            xml_bboxes[label].append([x1, y1, x2, y2])

        xml_path = os.path.join(output_folder, dataset_name + '_' + subfolder_name + '_' + img_name.replace('.jpg', '.xml'))
        write_xml(xml_path, img_path, xml_bboxes, img_shape)


def tranform(args):
    subfolder_list = os.listdir(args.input_dir)
    subfolder_list.sort()

    for subfolder_idx in tqdm(range(len(subfolder_list))):
        subfolder_name = subfolder_list[subfolder_idx]

        if subfolder_name in ignore_dataset:
            continue
        print(subfolder_name)

        subfolder_path = os.path.join( args.input_dir, subfolder_name ) 
        img_folder = os.path.join( subfolder_path, 'img1' ) 
        
        # train
        if args.dataset_type == "train":
            gt_path = os.path.join( subfolder_path, 'gt', 'gt.txt') 
        else:
            gt_path = os.path.join( subfolder_path, 'det', 'det.txt') 

        output_folder = os.path.join( args.output_dir, subfolder_name, 'JPEGImages' ) 
        create_folder(output_folder)

        label_to_xml(img_folder, gt_path, output_folder, args.dataset_name, subfolder_name, args)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.dataset_name = "MOT15"       # MOT15，行人被完全遮挡，仍然标注了，无法使用
    # args.dataset_name = "MOT16"       # MOT16，包含部分 MOT15 数据
    args.dataset_name = "MOT17"       # MOT17，包含 MOT16 数据
    # args.dataset_name = "MOT20"       # MOT20

    # args.dataset_type = "train"       # 训练集，标签
    args.dataset_type = "test"          # 测试集，检测效果不好

    args.input_dir = "/mnt/huanyuan2/data/image/ZG_BMX_detection/公开数据集/MOT/" + args.dataset_name + "/" + args.dataset_type + "/"
    args.output_dir = "/mnt/huanyuan2/data/image/ZG_BMX_detection/公开数据集/MOT/" + args.dataset_name + "/" + args.dataset_type + "_xml_gt/"
    args.person_visibility_threh = 0.3
    args.car_visibility_threh = 0.1

    tranform(args)
