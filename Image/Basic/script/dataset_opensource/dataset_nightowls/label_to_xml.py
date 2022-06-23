import argparse
import cv2
import os
from os import path
from tqdm import tqdm
import sys 

from pycocotools.coco import COCO

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
from Basic.script.xml.xml_write import write_xml
from Basic.utils.folder_tools import *


def tranform(args):

    output_folder = os.path.join( args.output_dir, 'JPEGImages' ) 
    create_folder(output_folder)

    cocoGt = COCO(args.json_ptah)
    imgIds = sorted(cocoGt.getImgIds())
    print('There are %d images in the training set' % len(imgIds))

    annotations = cocoGt.getAnnIds()
    print('There are %d annotations in the training set' % len(annotations))

    for idx in tqdm(range(len(imgIds))):

        img_id = imgIds[idx]
        annoIds = cocoGt.getAnnIds(imgIds=img_id)

        # # test 
        # image = cocoGt.loadImgs(ids=img_id)[0]
        # file_path = path.join(args.input_dir, image['file_name'])
        # if image['file_name'] == "58c58208bc2601370012d998.png":
        #     print()

        # xml
        xml_bboxes = {}
        for idy in range(len(annoIds)):
            anno_id = annoIds[idy]

            anno = cocoGt.loadAnns(ids=anno_id)[0]
            # print('Annotation (id=%d): %s' % (anno_id, anno))

            # cat
            cat = cocoGt.loadCats(ids=anno['category_id'])[0]
            category_name = cat['name']
            # print('Object type %s' % category_name)

            # bbox
            bbox = anno['bbox']
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
            w = bbox[2]
            h = bbox[3]
          
            # set class
            assert category_name in args.select_class_list
            category_idx = args.select_class_list.index(category_name)
            if w < args.width_threshold and h < args.height_threshold:
                category_name = args.filter_set_class_list[category_idx]
            else:
                category_name = args.set_class_list[category_idx]

            if category_name not in xml_bboxes:
                xml_bboxes[category_name] = []
            
            xml_bboxes[category_name].append([x1, y1, x2, y2])

        # check xml
        xml_bool = False
        for xml_key in xml_bboxes.keys():
            if xml_key in args.set_class_list:
                xml_bool = True
        if "ignore" in xml_bboxes.keys():
            xml_bool = False

        if xml_bool:
            image = cocoGt.loadImgs(ids=img_id)[0]
            file_path = path.join(args.input_dir, image['file_name'])
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            img_shape = [img.shape[1], img.shape[0], img.shape[2]]

            xml_path = os.path.join(output_folder, str(image['file_name']).replace('.png', '.xml'))
            write_xml(xml_path, file_path, xml_bboxes, img_shape)

            output_path = os.path.join(output_folder, str(image['file_name']).replace('.png', '.jpg'))
            cv2.imwrite(output_path, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.input_dir = "/mnt/huanyuan/temp/NightOwls/nightowls_training/"
    # args.json_ptah = "/mnt/huanyuan/temp/NightOwls/nightowls_training.json"
    # args.output_dir = "/mnt/huanyuan/temp/NightOwls/nightowls_training_xml_gt/"

    args.input_dir = "/mnt/huanyuan/temp/NightOwls/nightowls_validation/"
    args.json_ptah = "/mnt/huanyuan/temp/NightOwls/nightowls_validation.json"
    args.output_dir = "/mnt/huanyuan/temp/NightOwls/nightowls_validation_xml_gt/"

    args.select_class_list = ['pedestrian', 'bicycledriver', 'motorbikedriver', 'ignore']
    args.set_class_list = ["person", "bicyclist", "motorcyclist", 'ignore']
    args.filter_set_class_list = ["person_o", "bicyclist_o", "motorcyclist_o", 'ignore']
    args.width_threshold = 25
    args.height_threshold = 75
    tranform(args)