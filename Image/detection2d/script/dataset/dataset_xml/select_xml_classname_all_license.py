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

                    bbox_height = bndbox[3] - bndbox[1]

                    if bbox_height < args.filter_select_class_plate_height_threshold:
                        object.find('name').text = args.filter_set_class_plate

            if object.find('name').text in args.set_name_list:
                bool_have_pos = True
        
        # 删除无用标签
        for object in root.findall('object'):
            # name
            classname = str(object.find('name').text)

            if (args.filter_bool and classname not in args.set_name_list and classname not in args.filter_finnal_name_plate_list) or ( not args.filter_bool and classname not in args.set_name_list) :
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
            if (classname not in args.finnal_name_list and classname not in args.filter_finnal_name_plate_list):
                print(classname + "---->label is error---->" + classname)
        tree.write(output_xml_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ######################################
    # Annotations_License
    # ["license", "neg"]
    ######################################
    # RM_ADAS_AllInOne
    # 类别: "plate", "fuzzy_plate", "moto_plate", "mote_fuzzy_plate"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate/"
    # args.select_name_list = [ "plate", "fuzzy_plate", "moto_plate", "mote_fuzzy_plate" ]
    # args.set_name_list = [ "license", "license", "license", "license"]
    # args.finnal_name_list = [ "license", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "XML/"
    # # allinone_w_licenseplate: [('license', 17054), ('license_ignore', 39), ('neg', 4289)]

    # RM_ADAS_AllInOne
    # 类别:  "licence", "licence_o", "licence_f"（汽车车牌）
    # 注：allinone_w_licenseplate_2023_zph_new_style/ADAS_VehicleTailRegression 规则变动，人眼看不清字符的车牌都没有标注，不再适用
    # 注：allinone_w_licenseplate_2023_zph_new_style/ADAS_AllInOne_New_Test 无车牌标注
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0_5M_Backlight/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0_NearPerson/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_Backlight_AbnormalVehicle/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_MS1/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_Rainy_Night/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_Night_Highway_Backlight/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AbnormalVehicle/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_AbnormalVehicle/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_MS3_patch/"
    # ignore args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph_new_style/ADAS_AllInOne_New_Test/"
    # ignore args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph_new_style/ADAS_VehicleTailRegression/"
    # args.select_name_list = [ "licence", "licence_o", "licence_f"]
    # args.set_name_list = [ "license", "license", "license"]
    # args.finnal_name_list = [ "license", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "Annotations/"
    # allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0/: [('license', 1192), ('license_ignore', 187), ('neg', 843)]
    # allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0_5M_Backlight/: [('license', 2347), ('license_ignore', 837), ('neg', 795)]
    # allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0_NearPerson/: [('license', 960), ('license_ignore', 793), ('neg', 3791)]
    # allinone_w_licenseplate_2023_zph/ADAS_AllInOne_Backlight_AbnormalVehicle/: [('license', 3717), ('license_ignore', 3110), ('neg', 1558)]
    # allinone_w_licenseplate_2023_zph/ADAS_AllInOne_MS1/: [('license', 8865), ('license_ignore', 1243), ('neg', 1680)]
    # allinone_w_licenseplate_2023_zph/ADAS_AllInOne_Rainy_Night/: [('license', 6371), ('license_ignore', 2914), ('neg', 1106)]
    # allinone_w_licenseplate_2023_zph/ADAS_Night_Highway_Backlight/: [('license', 1302), ('license_ignore', 1261), ('neg', 2004)]
    # allinone_w_licenseplate_2023_zph/ADAS_AbnormalVehicle/: [('license', 12667), ('license_ignore', 9936), ('neg', 10797)]
    # allinone_w_licenseplate_2023_zph/ADAS_AllInOne_AbnormalVehicle: [('license', 6138), ('license_ignore', 1615), ('neg', 379)]
    # allinone_w_licenseplate_2023_zph/ADAS_AllInOne_MS3_patch/: [('license', 7968), ('license_ignore', 2959), ('neg', 774)]
    # allinone_w_licenseplate_2023_zph_new_style/ADAS_AllInOne_New_Test/: [('neg', 22294)]
    # allinone_w_licenseplate_2023_zph_new_style/ADAS_VehicleTailRegression/: [('license', 630), ('license_ignore', 267), ('neg', 7220)]

    # ZF_Europe
    # 类别: "car_bus_truck", "plate"（汽车车牌）
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/england/"
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/england_1080p/"
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/france/"
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/italy/"
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/netherlands/"
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/moni/"
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/moni_0415/"
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/hardNeg/"
    # args.select_name_list = ["licenseplate", "lecense", "lecense_f", "lecense_o", "licence", "licence_F", "licence_f", "licence_o"]
    # args.set_name_list = ["license", "license", "license", "license", "license", "license", "license", "license"]
    # args.finnal_name_list = ["license", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "XML/"
    # england: [('license', 56448), ('license_ignore', 16461), ('neg', 6886)]
    # england_1080p: [('license', 28358), ('license_ignore', 25164), ('neg', 9365)]
    # france: [('license', 65799), ('license_ignore', 20007), ('neg', 11627)]
    # italy: [('license', 9609), ('license_ignore', 1872), ('neg', 1017)]
    # netherlands: [('license', 42533), ('license_ignore', 19166), ('neg', 13656)]
    # moni: [('license', 14219), ('license_ignore', 797), ('neg', 2031)]
    # moni_0415: [('license', 5691), ('license_ignore', 583), ('neg', 8834)]
    # hardNeg: [('license', 11417), ('license_ignore', 17673), ('neg', 11401)]

    # ZG_ZHJYZ_detection
    # 注：shaobing 无车牌标注
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/jiayouzhan/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/jiayouzhan_5M/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/sandaofangxian/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/anhuihuaibeigaosu/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/shenzhentiaoqiao/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/anhuihuaibeigaosu_night_diguangzhao/"
    # ignore args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/shaobing/"
    # args.select_name_list = [ "plate", "fuzzy_plate", "roi_ignore_plate"]
    # args.set_name_list = [ "license", "license", "license" ]
    # args.finnal_name_list = [ "license", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "XML/"
    # jiayouzhan: [('license', 11601), ('license_ignore', 218), ('neg', 107)]
    # jiayouzhan_5M: [('license', 5172)]
    # sandaofangxian: [('license', 9949), ('license_ignore', 6), ('neg', 54)]
    # anhuihuaibeigaosu: [('license', 95038), ('license_ignore', 13417), ('neg', 668)]
    # shenzhentiaoqiao: [('license', 21606), ('license_ignore', 2640), ('neg', 168)]
    # anhuihuaibeigaosu_night_diguangzhao: [('license', 28492), ('license_ignore', 1618), ('neg', 7133)]
    # shaobing: [('neg', 11182)]

    # RM_C27_detection
    # args.input_dir = "/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil/Brazil_all/"
    # args.input_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_new_style/shate_20230308/"
    # args.select_name_list = [ "license", "cover-motorplate", "cover-plaet", "lince-plate", "fuzzy-plate", "lince-motorplate", "fuzzy-motorplate"]
    # args.set_name_list = [ "license", "license", "license", "license", "license", "license", "license"]
    # args.finnal_name_list = [ "license", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "Annotations/"
    # Brazil_all: [('license', 271046), ('license_ignore', 9008), ('neg', 70)]
    # shate_20230308: [('license', 38281), ('license_ignore', 1826), ('neg', 9)]

    # RM_C27_detection
    # 类别: "license_plate"
    # 注：license_plate 清晰车牌，不可用
    # ignore args.input_dir = "/yuanhuan/data/image/RM_C27_detection/original/zd_c27_20200209_20201125/"
    # args.select_name_list = [ "license_plate" ]
    # args.set_name_list = [ "license"]
    # args.finnal_name_list = [ "license", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "XML/"
    # zd_c27_20200209_20201125: [('license', 163109), ('license_ignore', 56), ('neg', 14047)]

    args.output_xml_dir =  args.input_dir + "Annotations_License/"

    ######################################
    # 大小阈值筛选
    ######################################

    args.filter_bool = True

    # 判断大小车牌
    args.filter_set_class_plate = "license_ignore"
    args.filter_finnal_name_plate_list = [ "license_ignore" ]
    args.filter_select_class_plate_height_threshold = 10

    select_classname(args)

    select_classname(args)

    # [ "license", "license_ignore", "neg"]