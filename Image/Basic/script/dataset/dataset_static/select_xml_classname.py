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

                        if bbox_width < args.width_threshold or bbox_height < args.height_threshold:
                            object.find('name').text = args.filter_set_class_list[filter_select_idx]
                    
            if object.find('name').text in args.set_name_list:
                bool_have_pos = True
        
        # 删除无用标签
        for object in root.findall('object'):
            classname = str(object.find('name').text)

            if (classname not in args.set_name_list):
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
            if not (classname in args.finnal_name_list):
                print(classname + "---->label is error---->" + classname)
        tree.write(output_xml_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ######################################
    # Annotations_CarBusTruckLicenseplate
    ######################################
    # # args.input_dir = "/yuanhuan/data/image/LicensePlate/China/"
    # # args.input_dir = "/yuanhuan/data/image/LicensePlate/China_6mm/"
    # # args.input_dir = "/yuanhuan/data/image/LicensePlate/Europe/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/Mexico/"
    # args.select_name_list = ["license_plate"]
    # args.set_name_list = ["license_plate"]
    # args.finnal_name_list = ["license_plate", "neg"]

    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone/"
    # args.select_name_list = ["car", "bus", "truck"]
    # args.set_name_list = ["car", "bus", "truck"]
    # args.finnal_name_list = ["car", "bus", "truck", "neg"]
        
    # args.jpg_dir =  args.input_dir + "JPEGImages/"
    # args.xml_dir =  args.input_dir + "XML/"
    # args.output_xml_dir =  args.input_dir + "Annotations_CarBusTruckLicenseplate/"

    ######################################
    # Annotations_CarNonMotorizedPerson
    ######################################
    # # RM_ADAS_AllInOne
    # # 类别: bicycle、motorcycle 表示没人骑行的数据，这里不参与训练
    # # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_new/"
    # args.select_name_list = ["car", "bus", "truck", "bicyclist", "motorcyclist", "person", "person_o", "preson"]
    # args.set_name_list = ["car", "bus", "truck", "bicyclist", "motorcyclist", "person", "person_o", "person"]
    # args.finnal_name_list = ["car", "bus", "truck", "bicyclist", "motorcyclist", "person", "person_o", "neg"]

    # # RM_BSD
    # # 类别: bicycle、motorcycle 表示没人骑行的数据，这里不参与训练
    # args.input_dir = "/yuanhuan/data/image/RM_BSD/bsd_20220425_20220512/"
    # args.select_name_list = ["car", "bus", "truck", "bicycle", "motorcycle", "person"]
    # args.set_name_list = ["car", "bus", "truck", "bicyclist", "motorcyclist", "person"]
    # args.finnal_name_list = ["car", "bus", "truck", "bicyclist", "motorcyclist", "person", "neg"]

    # ZG_BMX_detection
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/daminghu/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shandongyingzikou/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shandongyingzikou_night_hongwai/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/yongzou_night_hongwai/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shenzhenlukou/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shenzhenlukou_night_hongwai/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shenzhenlukou_night_diguangzhao/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/rongheng/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/rongheng_night_hongwai/"
    # args.select_name_list = ["car", "tricycle", "bus", "truck", "bicyclist", "motorcyclist", "person"]
    # args.set_name_list = ["car", "car", "bus", "truck", "bicyclist", "motorcyclist", "person"]
    # args.finnal_name_list = ["car", "car", "bus", "truck", "bicyclist", "motorcyclist", "person", "neg"]

    # 开源数据集: MOT17\MOT20
    # args.input_dir = "/yuanhuan/data/image/Open_Source/MOT/MOT17/"
    # args.select_name_list = ["car_bus_truck", "person"]
    # args.set_name_list = ["car", "person", "car_o", "person_o"]
    # args.finnal_name_list = ["car", "person", "car_o", "person_o", "neg"]

    # # 开源数据集: NightOwls
    # args.input_dir = "/yuanhuan/data/image/Open_Source/NightOwls/nightowls/"
    # args.select_name_list = ["person", "person_o"]
    # args.set_name_list = ["person", "person_o"]
    # args.finnal_name_list = ["person", "person_o", "neg"]
    
    # 开源数据集: Cityscapes
    args.input_dir = "/yuanhuan/data/image/Open_Source/Cityscapes/cityscapes/"
    args.select_name_list = ["car", "caravan", "bus", "train", "truck", "trailer", "bicyclist", "motorcyclist", "person"]
    args.set_name_list = ["car", "car", "bus", "bus", "truck", "truck", "bicyclist", "motorcyclist", "person", "car_o", "car_o", "bus_o", "bus_o", "truck_o", "truck_o", "bicyclist_o", "motorcyclist_o", "person_o"]
    args.finnal_name_list = ["car", "bus", "truck", "bicyclist", "motorcyclist", "person", "car_o", "car_o", "bus_o", "bus_o", "truck_o", "truck_o", "bicyclist_o", "motorcyclist_o", "person_o", "neg"]

    args.jpg_dir =  args.input_dir + "JPEGImages/"  
    args.xml_dir =  args.input_dir + "XML/"
    args.output_xml_dir =  args.input_dir + "Annotations_CarBusTruckBicyclistMotorcyclistPerson/"

    ######################################
    # 消融实验
    ######################################
    # Annotations_CarLicenseplate
    # China 提取 car\license_plate 进行实验
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/China/"
    # args.select_name_list = ["car", "license_plate"]
    # args.set_name_list = ["car", "license_plate"]
    # args.finnal_name_list = ["car", "license_plate", "neg"]

    # # # Mexico 提取 license_plate 参与 China 进行实验（验证 car 的性能不降低，同时 license_plate 性能提升）
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/Mexico/"
    # # # 训练过程中，不生成 car 的标签
    # # args.select_name_list = ["license_plate"]
    # # args.set_name_list = ["license_plate"]
    # # args.finnal_name_list = ["license_plate", "neg"]
    # # 测试过程中，生成 car 的标签，计算 AP 值
    # args.select_name_list = ["car", "license_plate"]
    # args.set_name_list = ["car", "license_plate"]
    # args.finnal_name_list = ["car", "license_plate", "neg"]

    # # China_6mm 提取 car\license_plate
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/China_6mm/"
    # args.select_name_list = ["car", "license_plate"]
    # args.set_name_list = ["car", "license_plate"]
    # args.finnal_name_list = ["car", "license_plate", "neg"]
        
    # args.jpg_dir =  args.input_dir + "JPEGImages/"
    # args.xml_dir =  args.input_dir + "XML/"
    # args.output_xml_dir =  args.input_dir + "Annotations_CarLicenseplate/"
    # # args.output_xml_dir =  args.input_dir + "Annotations_CarLicenseplate_label/"

    # ######################################
    # # 清洗标签，错误标签修正
    # ######################################

    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/sandaofangxian/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu/"

    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/"

    # args.select_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate", "painted_plate", "planted_plate"]
    # args.set_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate", "painted_plate", "painted_plate"]
    # args.finnal_name_list = ["car", "bus", "truck", "plate", "fuzzy_plate", "painted_plate"]

    # args.jpg_dir =  args.input_dir + "JPEGImages/"
    # args.xml_dir =  args.input_dir + "XML/"
    # args.output_xml_dir =  args.input_dir + "XML_refine/"

    # # args.jpg_dir =  args.input_dir + "SZTQ/"
    # # args.xml_dir =  args.input_dir + "SZTQ_XML/"
    # # args.output_xml_dir =  args.input_dir + "SZTQ_XML_refine/"
    
    ######################################
    # 大小阈值筛选
    ######################################

    # # args.filter_bool = False
    # args.filter_bool = True

    # # 2M
    # # args.width_threshold = 25
    # # args.height_threshold = 25

    # # 5M
    # args.width_threshold = 40
    # args.height_threshold = 40

    # args.filter_select_class_list = ["car", "caravan", "bus", "train", "truck", "trailer", "bicyclist", "motorcyclist", "person"]
    # args.filter_set_class_list = ["car_o", "car_o", "bus_o", "bus_o", "truck_o", "truck_o", "bicyclist_o", "motorcyclist_o", "person_o"]

    select_classname(args)