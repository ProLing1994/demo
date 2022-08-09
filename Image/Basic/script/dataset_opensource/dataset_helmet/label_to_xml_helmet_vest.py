import argparse
import cv2
import os
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
# sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.xml.xml_write import write_xml
from Basic.utils.folder_tools import *

zg_label_dict = { "0": '0',
                  "1": '1',
                  "2": '2',
                  "3": '3'
                }

def label_to_xml(img_folder, gt_path, output_folder, args):

    # gt
    gt_dict = {}
    with open(gt_path, "r") as f:
        for line in f:
            ine_str = line.strip().split('\t')
            img_name = ine_str[0]
            img_boxex = ine_str[1:]

            for bbox_idx in range(len( img_boxex )):
                bbox_list = img_boxex[bbox_idx].strip().split(',')
                bbox_ltrb = bbox_list[0:4]
                class_id = bbox_list[4]

                if img_name not in gt_dict:
                    gt_dict[img_name] = []
                    
                gt_dict[img_name].append({'img_name': img_name, 'bbox_ltrb':bbox_ltrb, 'class_id':class_id})

    # label to xml
    for img_idx in tqdm(gt_dict.keys()):

        # img
        img_name = img_idx
        img_path = os.path.join(img_folder, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_shape = [img.shape[1], img.shape[0], img.shape[2]]

        # xml
        xml_bboxes = {}
        for bbox_idx in range(len(gt_dict[img_idx])):
            gt_bbox_idx = gt_dict[img_idx][bbox_idx]

            label = zg_label_dict[gt_bbox_idx['class_id']]
            x1 = int(float(gt_bbox_idx['bbox_ltrb'][0]))
            y1 = int(float(gt_bbox_idx['bbox_ltrb'][1]))
            x2 = int(float(gt_bbox_idx['bbox_ltrb'][2]))
            y2 = int(float(gt_bbox_idx['bbox_ltrb'][3]))

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

        xml_path = os.path.join(output_folder, img_name.replace('.jpg', '.xml'))
        write_xml(xml_path, img_path, xml_bboxes, img_shape)


def tranform(args):

    create_folder(args.output_dir)

    img_folder = os.path.join( args.input_dir, 'Images' ) 
    label_folder = os.path.join( args.input_dir, 'Labels' ) 

    label_list = os.listdir( label_folder )
    label_list.sort()

    for label_idx in tqdm(range(len( label_list ))):
        label_path = os.path.join( label_folder, label_list[label_idx] ) 

        if label_list[label_idx].startswith( args.label_format ):
            label_to_xml(img_folder, label_path, args.output_dir, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/mnt/huanyuan/temp/helmet/helmet_vest/"
    args.output_dir = "/mnt/huanyuan/temp/helmet/helmet_vest/XML/"
    # args.label_format = "pictor_ppe_crowdsourced_approach-01"
    # args.label_format = "pictor_ppe_crowdsourced_approach-02"
    args.label_format = "pictor_ppe_crowdsourced_approach-03"

    tranform(args)