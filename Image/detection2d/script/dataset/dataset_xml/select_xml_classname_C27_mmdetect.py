import argparse
import numpy as np
import os
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
# sys.path.insert(0, '/yuanhuan/demo')
from Image.detection2d.script.dataset.dataset_xml.select_xml_classname_C27 import select_classname


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    #####################################
    # Annotations_CarBusTruckMotorcyclePlateMotoplate_w_fuzzy
    # ["car", "bus", "truck", "car_bus_truck"，"motorcyclist", "license_plate", "moto_license_plate", "neg"]
    #####################################
    # RM_ADAS_AllInOne
    # 类别: "car", "bus", "truck", "motorcyclist", "plate", "fuzzy_plate", "moto_plate", "mote_fuzzy_plate"
    # 注: motorcycle 表示没人骑行的数据，这里不参与训练
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate/"
    # args.select_name_list = ["car", "bus", "truck", "motorcyclist", "plate", "fuzzy_plate", "moto_plate", "mote_fuzzy_plate"]
    # args.set_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "license_plate", "license_plate", "license_plate"]
    # args.finnal_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "XML/"
    # 数据集: allinone_w_licenseplate, [('bus', 4488), ('car', 44139), ('license_plate', 17093), ('motorcyclist', 6073), ('neg', 460), ('truck', 2552)]
    
    # # RM_ADAS_AllInOne
    # # 类别: "car", "bus", "truck", "motorcyclist", "licence", "licence_f"（汽车车牌）
    # # 注: motorcycle 表示没人骑行的数据，这里不参与训练
    # # 注：allinone_w_licenseplate_2023_zph_new_style/ADAS_VehicleTailRegression 规则变动，人眼看不清字符的车牌都没有标注，不再适用
    # # 注：allinone_w_licenseplate_2023_zph_new_style/ADAS_AllInOne_New_Test 无车牌标注
    # # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0/"
    # # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0_5M_Backlight/"
    # # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0_NearPerson/"
    # # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_Backlight_AbnormalVehicle/"
    # # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_MS1/"
    # # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_Rainy_Night/"
    # # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_Night_Highway_Backlight/"
    # # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AbnormalVehicle/"
    # # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_AbnormalVehicle/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_MS3_patch/"
    # # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph_new_style/ADAS_AllInOne_New_Test/"
    # # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph_new_style/ADAS_VehicleTailRegression/"
    # args.select_name_list = ["car", "bus", "truck", "motorcyclist", "licence", "licence_f"]
    # args.set_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "license_plate"]
    # args.finnal_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "Annotations/"
    # 数据集: ADAS_ADPLUS2.0, [('bus', 189), ('car', 2885), ('license_plate', 1292), ('motorcyclist', 1097), ('neg', 101), ('truck', 372)]
    # 数据集: ADAS_ADPLUS2.0_5M_Backlight, [('bus', 309), ('car', 4153), ('license_plate', 3048), ('motorcyclist', 1582), ('neg', 70), ('truck', 428)]
    # 数据集: ADAS_ADPLUS2.0_NearPerson, [('car', 2653), ('license_plate', 1663), ('motorcyclist', 217), ('neg', 2480), ('truck', 21)]
    # 数据集: ADAS_AllInOne_Backlight_AbnormalVehicle, [('bus', 406), ('car', 9299), ('license_plate', 6758), ('motorcyclist', 1490), ('neg', 195), ('truck', 1700)]
    # 数据集: ADAS_AllInOne_MS1, [('bus', 1286), ('car', 19878), ('license_plate', 9638), ('motorcyclist', 2691), ('neg', 158), ('truck', 2456)]
    # 数据集: ADAS_AllInOne_Rainy_Night, [('bus', 295), ('car', 11394), ('license_plate', 9022), ('motorcyclist', 658), ('neg', 204), ('truck', 715)]
    # 数据集: ADAS_Night_Highway_Backlight, [('bus', 3), ('car', 2173), ('license_plate', 2541), ('neg', 93), ('truck', 3730)]
    # 数据集: ADAS_AbnormalVehicle, [('bus', 787), ('car', 62572), ('license_plate', 22342), ('motorcyclist', 753), ('neg', 76), ('truck', 25056)]
    # 数据集: ADAS_AllInOne_AbnormalVehicle, [('bus', 752), ('car', 9986), ('license_plate', 7533), ('motorcyclist', 404), ('neg', 1), ('truck', 4626)]
    # 数据集: ADAS_AllInOne_MS3_patch, [('bus', 1378), ('car', 16640), ('license_plate', 10482), ('motorcyclist', 2164), ('neg', 87), ('truck', 1998)]

    # # ZG_ZHJYZ_detection
    # # 注：shaobing 无车牌标注
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/jiayouzhan/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/jiayouzhan_5M/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/sandaofangxian/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/shenzhentiaoqiao/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/anhuihuaibeigaosu/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/anhuihuaibeigaosu_night_diguangzhao/"
    # # ignore args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/shaobing/"
    # args.select_name_list = ["car", "bus", "truck", "motorcyclist", "plate", "fuzzy_plate"]
    # args.set_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "license_plate"]
    # args.finnal_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "XML/"
    # 数据集: jiayouzhan, [('bus', 685), ('car', 12295), ('license_plate', 11592), ('motorcyclist', 44), ('truck', 180)]
    # 数据集: jiayouzhan_5M, [('bus', 489), ('car', 7774), ('license_plate', 5172), ('motorcyclist', 52), ('truck', 22)]\
    # 数据集: sandaofangxian, [('bus', 1784), ('car', 9226), ('license_plate', 9955), ('truck', 13267)]
    # 数据集: shenzhentiaoqiao, [('bus', 12175), ('car', 42484), ('license_plate', 40863), ('motorcyclist', 343), ('truck', 3097)]
    # 数据集: anhuihuaibeigaosu, [('bus', 2783), ('car', 231859), ('license_plate', 108455), ('truck', 46680)]
    # 数据集: anhuihuaibeigaosu_night_diguangzhao, [('bus', 3629), ('car', 80339), ('license_plate', 30110), ('truck', 11480)]
    
    # RM_C27_detection
    # args.input_dir = "/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil/Brazil_all/"
    # args.input_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_new_style/shate_20230308/"
    args.input_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_new_style/shate_20230309/"
    args.select_name_list = ["car", "bus", "truck", "motorcycle", "motorcyclist", "lince-plate", "fuzzy-plate", "lince-motorplate", "fuzzy-motorplate", "license"]
    args.set_name_list = ["car", "bus", "truck", "motorcyclist", "motorcyclist", "license_plate", "license_plate", "license_plate", "license_plate", "license_plate"]
    args.finnal_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "neg"]
    args.jpg_dir = args.input_dir + "JPEGImages/"  
    args.xml_dir = args.input_dir + "Annotations/"
    # 数据集: Brazil_all, [('bus', 15165), ('car', 433038), ('license_plate', 239920), ('motorcyclist', 58671), ('neg', 7), ('truck', 34930)]
    # 数据集: shate_20230308, [('bus', 1205), ('car', 89409), ('license_plate', 39818), ('motorcyclist', 98), ('truck', 7082)]
    # 数据集: shate_20230309, [('bus', 344), ('car', 35150), ('license_plate', 17595), ('motorcyclist', 111), ('truck', 4946)]

    args.output_xml_dir =  args.input_dir + "Annotations_CarBusTruckMotorcyclePlate_MMDetect/"

    ######################################
    # 大小阈值筛选
    ######################################

    args.filter_bool = False

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

    # ["car", "bus", "truck", "motorcyclist", "license_plate", "neg"]