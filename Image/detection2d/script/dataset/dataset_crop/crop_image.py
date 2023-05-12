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
    jpg_list.sort()

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

                x1, x2 = bndbox[0], bndbox[2]
                y1, y2 = bndbox[1], bndbox[3]
                
                if args.padding_bool:

                    width = x2 - x1
                    height = y2 - y1

                    center_x = int((x1 + x2)/2 + 0.5)
                    center_y = int((y1 + y2)/2 + 0.5)

                    if width < args.min_size:
                        x1 = max(0, int(center_x - args.min_size / 2))
                        x2 = min(img.shape[1], int(center_x + args.min_size / 2))

                    if height < args.min_size:
                        y1 = max(0, int(center_y - args.min_size / 2))
                        y2 = min(img.shape[0], int(center_y + args.min_size / 2))

                crop_img = img[y1:y2, x1:x2]

                output_img_path = os.path.join(args.output_dir, name, jpg_list[idx].replace(".jpg", "_{}.jpg".format(idy)))

                # mkdir 
                if not os.path.exists( os.path.dirname(output_img_path) ):
                    os.makedirs( os.path.dirname(output_img_path) )

                # continue
                if os.path.exists( output_img_path ):
                    continue
                
                try:
                    cv2.imwrite(output_img_path, crop_img)
                except:
                    print(output_img_path)
                idy += 1


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.input_dir = "/mnt/huanyuan2/data/image/LicensePlate/Mexico/"
    # args.select_name_list = ["license_plate"]
    # args.jpg_dir =  args.input_dir + "JPEGImages/"
    # args.xml_dir =  args.input_dir + "XML/"
    # args.output_dir = args.input_dir + "Crop/"

    # # args.input_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan/"
    # # args.input_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_5M/"
    # # args.input_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/sandaofangxian/"
    # # args.input_dir = "/mnt/huanyuan2/data/image/ZG_AHHBGS_detection/anhuihuaibeigaosu/"
    # # args.input_dir = "/mnt/huanyuan2/data/image/ZG_AHHBGS_detection/shenzhentiaoqiao/"
    # args.input_dir = "/mnt/huanyuan2/data/image/ZG_AHHBGS_detection/20220530_新增数据/"
    # args.select_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate", 'painted_plate', "roi_ignore_plate"]
    # args.jpg_dir =  args.input_dir + "JPEGImages/"
    # args.xml_dir =  args.input_dir + "XML/"
    # args.output_dir = args.input_dir + "Crop/"

    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/daminghu/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/daminghu_night/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/rongheng/"
    args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/rongheng_night_hongwai/"
    args.select_name_list = ["motorcyclist", "bicyclist"]
    args.jpg_dir =  args.input_dir + "JPEGImages_test/"
    args.xml_dir =  args.input_dir + "XML/"
    args.output_dir = args.input_dir + "Crop_Moto/"

    args.padding_bool = True
    args.min_size = 320

    crop_image(args)


if __name__ == '__main__':
    main()