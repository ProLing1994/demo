import argparse
import cv2
import os
from os import path
from tqdm import tqdm
import sys 

# sys.path.insert(0, '/home/huanyuan/code/demo/Image')
sys.path.insert(0, '/yuanhuan/code/demo/Image/')
from Basic.script.xml.xml_write import write_xml
from Basic.utils.folder_tools import *


def tranform(args):
    
    # file
    with open(args.input_txt_path, "r") as f:
        
        # init
        file_name_line, num_boxes_line, box_annotation_line = True, False, False

        img_path = ""
        num_boxes, box_counter, idx = 0, 0, 0
        labels = []

        lines = f.readlines()
        progress_bar = tqdm(lines)
        for line in progress_bar:

            line = line.rstrip()

            if file_name_line:

                img_path = line
                file_name_line = False
                num_boxes_line = True

            elif num_boxes_line:

                num_boxes = int(line)
                num_boxes_line = False
                box_annotation_line = True
            
            elif box_annotation_line:

                box_counter += 1

                line_split = line.split(" ")
                line_values = [int(x) for x in line_split]
                labels.append(line_values)

                if box_counter >= num_boxes:

                    box_annotation_line = False
                    file_name_line = True
                    
                    # img
                    img = cv2.imread(os.path.join(args.output_img_dir, img_path))
                    img_shape = [img.shape[1], img.shape[0], img.shape[2]]

                    # xml
                    xml_bboxes = {}

                    if len(labels) == 0:
                        xml_bboxes['neg'] = []
                        xml_bboxes['neg'].append([10, 10, 10, 10])

                    for label_idx in range(len(labels)):
                        
                        label = labels[label_idx]

                        x1 = label[0]
                        y1 = label[1]
                        w = label[2]
                        h = label[3]
                        x2 = x1 + w
                        y2 = y1 + h

                        blur = label[4]
                        expression = label[5]
                        illumination = label[6]
                        invalid = label[7]
                        occlusion = label[8]
                        pose = label[9]

                        class_name = args.set_class_name

                        if w < args.width_threshold or h < args.height_threshold or invalid == 1 or occlusion == 2:
                            class_name = args.filter_set_class_name

                        if class_name not in xml_bboxes:
                            xml_bboxes[class_name] = []

                        xml_bboxes[class_name].append([x1, y1, x2, y2])

                    xml_path = os.path.join(args.output_xml_dir, img_path.replace('.jpg', '.xml'))
                    create_folder(os.path.dirname(xml_path))
                    write_xml(xml_path, img_path, xml_bboxes, img_shape)
                    
                    box_counter = 0
                    labels.clear()
                    idx += 1
                    progress_bar.set_description(f"{idx} images")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.set_class_name = "face"
    args.filter_set_class_name = "face_o"

    args.width_threshold = 10
    args.height_threshold = 10

    args.input_txt_path = "/yuanhuan/data/image/Open_Source/Wider_Face/original/wider_face_split/wider_face_val_bbx_gt.txt"
    args.output_dir = "/yuanhuan/data/image/Open_Source/Wider_Face/original/WIDER_val/"
    args.output_img_dir = os.path.join(args.output_dir, "images")
    args.output_xml_dir = os.path.join(args.output_dir, "xml")

    tranform(args)    

    args.input_txt_path = "/yuanhuan/data/image/Open_Source/Wider_Face/original/wider_face_split/wider_face_train_bbx_gt.txt"
    args.output_dir = "/yuanhuan/data/image/Open_Source/Wider_Face/original/WIDER_train/"
    args.output_img_dir = os.path.join(args.output_dir, "images")
    args.output_xml_dir = os.path.join(args.output_dir, "xml")

    tranform(args)    