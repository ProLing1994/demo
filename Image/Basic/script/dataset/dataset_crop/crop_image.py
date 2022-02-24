import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET


def crop_image(args):
    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]

    for idx in tqdm(range(len(jpg_list))):
        jpg_apth = os.path.join(args.jpg_dir, jpg_list[idx])
        xml_path = os.path.join(args.xml_dir, jpg_list[idx].replace(".jpg", ".xml"))

        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点
        
        # 读取标签
        idy = 0
        for obj in root.iter('object'):
            # name
            name = obj.find('name').text.lower().strip()

            # bbox
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                bndbox.append(cur_pt)

            if name in args.select_name_list:
                
                # 读取图像
                img = cv2.imread(jpg_apth, cv2.IMREAD_COLOR)

                crop_img = img[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]]

                output_img_path = os.path.join(args.output_dir, name, jpg_list[idx].replace(".jpg", "_{}.jpg".format(idy)))

                # mkdir 
                if not os.path.exists( os.path.dirname(output_img_path) ):
                    os.makedirs( os.path.dirname(output_img_path) )

                cv2.imwrite(output_img_path, crop_img)
                idy += 1


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.input_dir = "/mnt/huanyuan2/data/image/LicensePlate/Mexico/"
    # args.select_name_list = ["license_plate"]
    # args.jpg_dir =  args.input_dir + "JPEGImages/"
    # args.xml_dir =  args.input_dir + "XML/"
    # args.output_dir = args.input_dir + "Crop/"

    args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan/"
    # args.select_name_list = ["plate", "fuzzy_plate"]
    args.select_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate"]
    args.jpg_dir =  args.input_dir + "JPEGImages/"
    args.xml_dir =  args.input_dir + "XML/"
    args.output_dir = args.input_dir + "Crop/"

    # args.input_dir = "/mnt/huanyuan2/data/image/RM_ADAS_AllInOne/allinone_lincense_plate/"
    # args.select_name_list = ["plate", "fuzzy_plate", "painted_plate"]
    # args.jpg_dir =  args.input_dir + "JPEGImages/"
    # args.xml_dir =  args.input_dir + "XML/"
    # args.output_dir = args.input_dir + "Crop/"

    crop_image(args)


if __name__ == '__main__':
    main()