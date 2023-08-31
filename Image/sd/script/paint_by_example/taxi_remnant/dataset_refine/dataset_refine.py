import argparse
import cv2
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *
from Basic.script.xml.xml_write import write_xml

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *
from Basic.script.xml.xml_write import write_xml
from sd.script.paint_by_example.taxi_remnant.dataset.prepare_dataset import center_crop

def dataset_refine(args):

    # mkdir 
    create_folder(args.output_img_dir)
    create_folder(args.output_xml_dir)
    create_folder(args.output_xml_refine_dir)

    # jpg list
    jpg_list = np.array(os.listdir(args.input_img_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.input_xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]
    jpg_list.sort()

    # refine grid list
    refine_grid_select_no_bottle_list = os.listdir(os.path.join(args.refine_grid_select_dir, 'no_bottle'))
    refine_grid_select_success_list = os.listdir(os.path.join(args.refine_grid_select_dir, 'success'))
    refine_grid_select_fail_list = os.listdir(os.path.join(args.refine_grid_select_dir, 'fail'))

    for idx in tqdm(range(len(jpg_list))):

        img_path = os.path.join(args.input_img_dir, jpg_list[idx])
        xml_path = os.path.join(args.input_xml_dir, jpg_list[idx].replace(".jpg", ".xml"))
        refine_img_path = os.path.join(args.refine_img_dir, jpg_list[idx])
        output_img_path = os.path.join(args.output_img_dir, jpg_list[idx])
        output_xml_path = os.path.join(args.output_xml_dir, jpg_list[idx].replace(".jpg", ".xml"))
        output_xml_refine_path = os.path.join(args.output_xml_refine_dir, jpg_list[idx].replace(".jpg", ".xml"))

        # img
        img = cv2.imread(img_path)

        # xml
        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点
        xml_bboxes = {}
        xml_refine_bboxes = {}

        # id 
        id = 0

        # bool
        bool_draw = False

        # 标签检测和标签转换
        for object in root.findall('object'):
            # name
            classname = str(object.find('name').text)

            # bbox
            bbox = object.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                bndbox.append(cur_pt)
            bndbox[0] = max(0, bndbox[0])
            bndbox[1] = max(0, bndbox[1])
            bndbox[2] = min(img.shape[1], bndbox[2])
            bndbox[3] = min(img.shape[0], bndbox[3])

            if classname not in args.infer_key_list:
                continue

            # 获得 Crop Img，以及对应坐标点
            if args.crop_bool:
                # center_crop
                roi_crop, size_crop = center_crop(args, img, bndbox)
            
            # check
            grid_name = str(jpg_list[idx]).replace('.jpg', '_{}.jpg'.format(id))

            if grid_name in refine_grid_select_no_bottle_list:
                # xml
                if args.error_key not in xml_refine_bboxes:
                    xml_refine_bboxes[args.error_key] = []   
                xml_refine_bboxes[args.error_key].append([bndbox[0], bndbox[1], bndbox[2], bndbox[3]])

                # xml
                if classname not in xml_bboxes:
                    xml_bboxes[classname] = []   
                xml_bboxes[classname].append([bndbox[0], bndbox[1], bndbox[2], bndbox[3]])

            elif grid_name in refine_grid_select_success_list:

                # xml
                if classname not in xml_refine_bboxes:
                    xml_refine_bboxes[classname] = []   
                xml_refine_bboxes[classname].append([bndbox[0], bndbox[1], bndbox[2], bndbox[3]])

                # xml
                if classname not in xml_bboxes:
                    xml_bboxes[classname] = []   
                xml_bboxes[classname].append([bndbox[0], bndbox[1], bndbox[2], bndbox[3]])

                # refine_img
                refine_img = cv2.imread(refine_img_path)
                img[roi_crop[1]:roi_crop[3], roi_crop[0]:roi_crop[2]] = refine_img[roi_crop[1]:roi_crop[3], roi_crop[0]:roi_crop[2]]

                bool_draw = True

            elif grid_name in refine_grid_select_fail_list:

                # xml
                if classname not in xml_refine_bboxes:
                    xml_refine_bboxes[classname] = []   
                xml_refine_bboxes[classname].append([bndbox[0], bndbox[1], bndbox[2], bndbox[3]])

                # xml
                if classname not in xml_bboxes:
                    xml_bboxes[classname] = []   
                xml_bboxes[classname].append([bndbox[0], bndbox[1], bndbox[2], bndbox[3]])
            
            else:
                raise Exception
            
            id += 1


        if bool_draw:
            # img
            cv2.imwrite(output_img_path, img)
            # xml
            write_xml(output_xml_path, output_img_path, xml_bboxes, img.shape)

        # # xml
        # write_xml(output_xml_refine_path, img_path, xml_refine_bboxes, img.shape)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="bottle") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/Taxi_remnant/original_select/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/Taxi_remnant/sd/paint_by_example/select_sam_bottle_02/") 
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.date_name)
    args.output_dir = os.path.join(args.output_dir, args.date_name)

    print("dataset refine.")
    print("date_name: {}".format(args.date_name))
    print("input_dir: {}".format(args.input_dir))
    print("output_dir: {}".format(args.output_dir))

    args.input_img_dir = os.path.join(args.input_dir, 'image')
    args.input_xml_dir = os.path.join(args.input_dir, 'xml')
    args.refine_img_dir = os.path.join(args.output_dir, 'JPEGImages')
    args.refine_grid_dir = os.path.join(args.output_dir, 'Grids')
    args.refine_grid_select_dir = os.path.join(args.output_dir, 'Grids_select')
    args.output_img_dir = os.path.join(args.output_dir, 'JPEGImages_select')
    args.output_xml_dir = os.path.join(args.output_dir, 'Annotations_select')
    args.output_xml_refine_dir = os.path.join(args.input_dir, 'xml_refine')

    # select
    args.infer_key_list = ['remnants']
    args.error_key = 'remnants_ignore'

    # crop
    args.crop_bool = True
    args.crop_size_list = [(64, 64), (128, 128), (256, 256), (512, 512), (720, 720)]    # w, h

    dataset_refine(args)