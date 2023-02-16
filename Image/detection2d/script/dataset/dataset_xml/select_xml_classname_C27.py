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
                                (img_width == 640 and img_height == 1024):
                            if bbox_width < args.width_threshold_2M or bbox_height < args.height_threshold_2M:
                                object.find('name').text = args.filter_set_class_list[filter_select_idx]
                        elif ((img_width == 1280 and img_height == 720 or (img_width == 704 and img_height == 576 ))):
                            if bbox_width < args.width_threshold_720p or bbox_height < args.height_threshold_720p:
                                object.find('name').text = args.filter_set_class_list[filter_select_idx]
                        else:
                            raise EOFError

                    if classname in args.filter_select_class_plate_list:
                        filter_select_idx = args.filter_select_class_plate_list.index(classname)

                        if bbox_height < args.filter_select_class_plate_height_threshold:
                            object.find('name').text = args.filter_set_class_plate_list[filter_select_idx]

            if object.find('name').text in args.set_name_list:
                bool_have_pos = True
        
        # 删除无用标签
        for object in root.findall('object'):
            classname = str(object.find('name').text)

            if (args.filter_bool and classname not in args.set_name_list and classname not in args.filter_set_class_list and classname not in args.filter_finnal_name_plate_list) or ( not args.filter_bool and classname not in args.set_name_list) :
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
            if (classname not in args.finnal_name_list and classname not in args.filter_finnal_name_list and classname not in args.filter_finnal_name_plate_list):
                print(classname + "---->label is error---->" + classname)
        tree.write(output_xml_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ######################################
    # Annotations_CarBusTruckMotorcyclePlateMotoplate_w_fuzzy
    # ["car", "bus", "truck", "car_bus_truck"，"motorcyclist", "license_plate", "moto_license_plate", "neg"]
    ######################################
    # RM_ADAS_AllInOne
    # 类别："car", "bus", "truck", "motorcyclist"
    # 注: motorcycle 表示没人骑行的数据，这里不参与训练
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_new/"
    # args.select_name_list = ["car", "bus", "truck", "motorcyclist"]
    # args.set_name_list = ["car", "bus", "truck", "motorcyclist"]
    # args.finnal_name_list = ["car", "bus", "truck", "motorcyclist", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "XML/"

    # RM_ADAS_AllInOne
    # 类别: "car", "bus", "truck", "motorcyclist", "plate", "fuzzy_plate", "moto_plate", "mote_fuzzy_plate"
    # 注: motorcycle 表示没人骑行的数据，这里不参与训练
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate/"
    # args.select_name_list = ["car", "bus", "truck", "motorcyclist", "plate", "fuzzy_plate", "moto_plate", "mote_fuzzy_plate"]
    # args.set_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "license_plate", "moto_license_plate", "moto_license_plate"]
    # args.finnal_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "moto_license_plate", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "XML/"

    # RM_ADAS_AllInOne
    # 类别: "car", "bus", "truck", "motorcyclist", "licence", "licence_f"（汽车车牌）
    # 注: motorcycle 表示没人骑行的数据，这里不参与训练
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0_5M_Backlight/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0_NearPerson/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_Backlight_AbnormalVehicle/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_MS1/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_New_Test/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_Rainy_Night/"
    args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_Night_Highway_Backlight/"
    args.select_name_list = ["car", "bus", "truck", "motorcyclist", "licence", "licence_f"]
    args.set_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "license_plate"]
    args.finnal_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "neg"]
    args.jpg_dir = args.input_dir + "JPEGImages/"  
    args.xml_dir = args.input_dir + "Annotations/"

    # # ZF_Europe
    # # 类别: "car_bus_truck", "plate"（汽车车牌）
    # # args.input_dir = "/yuanhuan/data/image/ZF_Europe/england/"
    # # args.input_dir = "/yuanhuan/data/image/ZF_Europe/england_1080p/"
    # # args.input_dir = "/yuanhuan/data/image/ZF_Europe/france/"
    # # args.input_dir = "/yuanhuan/data/image/ZF_Europe/italy/"
    # # args.input_dir = "/yuanhuan/data/image/ZF_Europe/netherlands/"
    # # args.input_dir = "/yuanhuan/data/image/ZF_Europe/moni/"
    # # args.input_dir = "/yuanhuan/data/image/ZF_Europe/moni_0415/"
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/hardNeg/"
    # args.select_name_list = ["car_bus_truck", "plate", "fuzzy_plate"]
    # args.set_name_list = ["car_bus_truck", "license_plate", "license_plate"]
    # args.finnal_name_list = ["car_bus_truck", "license_plate", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "XML_refine/"

    # ZG_BMX_detection
    # 类别: "car", "tricycle", "bus", "truck", "motorcyclist"
    # # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/daminghu/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/daminghu_night/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shandongyingzikou/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shandongyingzikou_night_hongwai/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/yongzou_night_hongwai/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shenzhenlukou/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shenzhenlukou_night_hongwai/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shenzhenlukou_night_diguangzhao/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/rongheng/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/rongheng_night_hongwai/"
    # args.select_name_list = ["car", "tricycle", "bus", "truck", "motorcyclist"]
    # args.set_name_list = ["car", "car", "bus", "truck", "motorcyclist"]
    # args.finnal_name_list = ["car", "bus", "truck", "motorcyclist", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "XML/"

    # # RM_C27_detection
    # # 类别: "car", "license_plate"（汽车车牌）
    # args.input_dir = "/yuanhuan/data/image/RM_C27_detection/zd_c27_2020_0209_1125/"
    # args.select_name_list = ["car", "license_plate"]
    # args.set_name_list = ["car_bus_truck", "license_plate"]
    # args.finnal_name_list = ["car_bus_truck", "license_plate", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "XML/"

    args.output_xml_dir =  args.input_dir + "Annotations_CarBusTruckMotorcyclePlateMotoplate_w_fuzzy/"

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

    args.filter_select_class_list = ["car", "tricycle", "bus", "truck", "car_bus_truck", "motorcyclist"]
    args.filter_set_class_list = ["car_o", "car_o", "bus_o", "truck_o", "car_bus_truck_o", "motorcyclist_o"]
    args.filter_finnal_name_list = ["car_o", "bus_o", "truck_o", "car_bus_truck_o", "motorcyclist_o"]

    # 判断大小车牌
    args.filter_select_class_plate_list = ["license_plate", "licence", "licence_f", 'plate', "fuzzy_plate", "moto_plate", "mote_fuzzy_plate"]
    args.filter_set_class_plate_list = ['license_plate_ignore', 'license_plate_ignore', 'license_plate_ignore', 'license_plate_ignore', "license_plate_ignore", "moto_license_plate_ignore", "moto_license_plate_ignore"]
    args.filter_finnal_name_plate_list = ["license_plate_ignore", "moto_license_plate_ignore"]
    args.filter_select_class_plate_height_threshold = 10

    select_classname(args)

    # ["car", "bus", "truck", "motorcyclist", "license_plate", "moto_license_plate", "car_bus_truck","neg"]