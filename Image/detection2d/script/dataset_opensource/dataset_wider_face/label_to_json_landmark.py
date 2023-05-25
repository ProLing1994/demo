import argparse
import cv2
import os
from os import path
from tqdm import tqdm
import sys 

# sys.path.insert(0, '/home/huanyuan/code/demo/Image')
sys.path.insert(0, '/yuanhuan/code/demo/Image/')
from Basic.script.json.json_write import write_json
from Basic.utils.folder_tools import *


def tranform(args):

    imgs_path = []
    words = []

    # file
    with open(args.input_txt_path, "r") as f:

        lines = f.readlines()

        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
        
    pbar = tqdm(zip(imgs_path, words), desc='Scanning images', total=len(imgs_path))
    for i, (im, lb) in enumerate(pbar):
        
        # img
        img = cv2.imread(os.path.join(args.output_img_dir, im))
        
        # json
        json_bboxes = {}

        for idx, label in enumerate(lb):
            
            if args.tarin_bool:

                w = label[2]
                h = label[3]

                points_list = []
                points_list.append([label[0], label[1]])    # x1 y1
                points_list.append([label[0] + label[2], label[1] + label[3]])    # x2 y2

                # landmarks
                points_list.append([label[4], label[5]])    # l0_x l0_y

                points_list.append([label[7], label[8]])    # l1_x l1_y

                points_list.append([label[10], label[11]])    # l2_x l2_y

                points_list.append([label[13], label[14]])    # l3_x l3_y 

                points_list.append([label[16], label[17]])    # l4_x l4_y
            else:

                w = label[2]
                h = label[3]

                points_list = []
                points_list.append([label[0], label[1]])    # x1 y1
                points_list.append([label[0] + label[2], label[1] + label[3]])    # x2 y2

                # landmarks
                points_list.append([-1.0, -1.0])    # l0_x l0_y

                points_list.append([-1.0, -1.0])    # l1_x l1_y

                points_list.append([-1.0, -1.0])    # l2_x l2_y

                points_list.append([-1.0, -1.0])    # l3_x l3_y 

                points_list.append([-1.0, -1.0])    # l4_x l4_y

            # class_name
            class_name = args.set_class_name

            if w < args.width_threshold or h < args.height_threshold:
                class_name = args.filter_set_class_name
                
            if class_name not in json_bboxes:
                json_bboxes[class_name] = []     
            json_bboxes[class_name].append(points_list)
        
        # output 
        output_json_path = os.path.join(args.output_json_dir, im.replace('.jpg', '.json'))
        create_folder(os.path.dirname(output_json_path))
        write_json(output_json_path, im, img.shape, json_bboxes, "polygon")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.set_class_name = "face"
    args.filter_set_class_name = "face_o"

    args.width_threshold = 10
    args.height_threshold = 10

    # args.input_txt_path = "/yuanhuan/data/image/Open_Source/Wider_Face/original/retinaface_gt_v1.1/train/label.txt"
    # args.output_dir = "/yuanhuan/data/image/Open_Source/Wider_Face/original/WIDER_train/"
    # args.output_img_dir = os.path.join(args.output_dir, "images")
    # args.output_json_dir = os.path.join(args.output_dir, "json_landmark")
    # args.tarin_bool = True

    # tranform(args)    

    args.input_txt_path = "/yuanhuan/data/image/Open_Source/Wider_Face/original/retinaface_gt_v1.1/val/label.txt"
    args.output_dir = "/yuanhuan/data/image/Open_Source/Wider_Face/original/WIDER_val/"
    args.output_img_dir = os.path.join(args.output_dir, "images")
    args.output_json_dir = os.path.join(args.output_dir, "json_landmark")
    args.tarin_bool = False

    tranform(args)    