import argparse
import numpy as np
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET


def select_classname(args):
    # mkdir 
    if not os.path.exists( args.output_xml_dir ):
        os.makedirs( args.output_xml_dir )

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]

    for idx in tqdm(range(len(jpg_list))):
        xml_path = os.path.join(args.xml_dir, jpg_list[idx].replace(".jpg", ".xml"))
        output_xml_path = os.path.join(args.output_xml_dir, jpg_list[idx].replace(".jpg", ".xml"))

        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

        bool_have_pos = False       # 检测是否包含正样本标注
        bool_have_one_neg = False   # 检测是否包含一个负样本（没有正样本时生效，保证存在一个标签）
        
        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)
        # 标签检测和标签转换
        for object in root.findall('object'):
            # name
            classname = str(object.find('name').text.lower().strip())

            # bbox
            bbox = object.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                bndbox.append(cur_pt)

            # 检测是否在 args.select_name_list 中
            if classname in args.select_name_list:
                select_idx = args.select_name_list.index(classname)
                object.find('name').text = args.set_name_list[select_idx]

                # 是否过滤
                if args.filter_bool == True:

                    bbox_width = bndbox[2] - bndbox[0]
                    bbox_height = bndbox[3] - bndbox[1]

                    if classname in args.filter_select_class_list:
                        filter_select_idx = args.filter_select_class_list.index(classname)

                        if (img_width == 2592 and img_height == 1920) or \
                            (img_width == 1024 and img_height == 2048):
                            if bbox_width < args.width_threshold_5M or bbox_height < args.height_threshold_5M:
                                object.find('name').text = args.filter_set_class_list[filter_select_idx]
                        elif (img_width == 1920 and img_height == 1080) or \
                                (img_width == 1080 and img_height == 1920) or \
                                (img_width == 480 and img_height == 640) or \
                                (img_width == 640 and img_height == 1024) or \
                                (img_height == 1080):
                            if bbox_width < args.width_threshold_2M or bbox_height < args.height_threshold_2M:
                                object.find('name').text = args.filter_set_class_list[filter_select_idx]
                        elif (img_width == 1280 and img_height == 720) or \
                                (img_width == 704 and img_height == 576 ) or \
                                (img_width == 900 and img_height == 1300 ) or \
                                (img_width == 720 and img_height == 1280 ) or \
                                (img_height == 720) :
                            if bbox_width < args.width_threshold_720p or bbox_height < args.height_threshold_720p:
                                object.find('name').text = args.filter_set_class_list[filter_select_idx]
                        else:
                            print("img_width: ", img_width)
                            print("img_height: ", img_height)

                            raise EOFError

            if object.find('name').text in args.set_name_list:
                bool_have_pos = True
        
        # 删除无用标签
        for object in root.findall('object'):
            classname = str(object.find('name').text)

            if (args.filter_bool and classname not in args.set_name_list and classname not in args.filter_set_class_list ) or ( not args.filter_bool and classname not in args.set_name_list) :
                # 如果存在正样本，删除无用标签
                if bool_have_pos:
                    root.remove(object)
                # 如果没有正样本时，生成唯一一个负样本标签（用于训练过程中的负样本）
                else:
                    print("have special_class neg file:", jpg_list[idx], "class_name is: ", classname)
                    if not bool_have_one_neg:
                        object.find('name').text = "neg"
                        bndbox = object.find('bndbox')
                        bndbox.find('xmin').text = "1900"
                        bndbox.find('ymin').text = "0"
                        bndbox.find('xmax').text = "1919"
                        bndbox.find('ymax').text = "19"
                        bool_have_one_neg = True
                    # 如果存在负样本，则不再生成标签
                    else:
                        root.remove(object)
            
        # 检测标签
        for object in root.findall('object'):
            classname = str(object.find('name').text)
            if (classname not in args.finnal_name_list and classname not in args.filter_finnal_name_list ):
                print(classname + "---->label is error---->" + classname)
        tree.write(output_xml_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ######################################
    # Annotations_Car
    # ["car", "neg"]
    ######################################
    # # 数据集: RM_BSD
    # # 类别："car", "bus", "truck"
    # # 注: motorcycle 表示没人骑行的数据，这里不参与训练
    # # args.input_dir = "/yuanhuan/data/image/RM_BSD/bsd_20220425_20220512/"
    # args.input_dir = "/yuanhuan/data/image/RM_BSD/wideangle_2022_2023/"
    # args.select_name_list = ["car", "bus", "truck"]
    # args.set_name_list = ["car", "car", "car"]
    # args.finnal_name_list = ["car", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # # args.xml_dir = args.input_dir + "XML/"
    # args.xml_dir = args.input_dir + "Annotations/"

    # 数据集: RM_C28_detection
    # 类别: "car"
    # args.input_dir = "/yuanhuan/data/image/RM_C28_detection/zhongdong/"
    # args.input_dir = "/yuanhuan/data/image/RM_C28_detection/safezone/"
    # args.input_dir = "/yuanhuan/data/image/RM_C28_detection/finished/"
    # args.input_dir = "/yuanhuan/data/image/RM_C28_detection/canada/"
    args.input_dir = "/yuanhuan/data/image/RM_C28_detection/america/"
    args.select_name_list = ["car", "CAR", "car "]
    args.set_name_list = ["car", "car", "car"]
    args.finnal_name_list = ["car", "neg"]
    args.jpg_dir = args.input_dir + "JPEGImages/"  
    args.xml_dir = args.input_dir + "Annotations/"

    args.output_xml_dir =  args.input_dir + "Annotations_Car/"

    ######################################
    # 大小阈值筛选
    ######################################

    # args.filter_bool = False
    args.filter_bool = True

    # 720p
    args.width_threshold_720p = 15
    args.height_threshold_720p = 15

    # 2M
    args.width_threshold_2M = 25
    args.height_threshold_2M = 25

    # 5M
    args.width_threshold_5M = 40
    args.height_threshold_5M = 40

    args.filter_select_class_list = ["car"]
    args.filter_set_class_list = ["car_o"]
    args.filter_finnal_name_list = ["car_o"]

    select_classname(args)

    # ["car", "neg"]