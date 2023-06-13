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
                            pass

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
    # 数据集: allinone, [('bus', 48057), ('bus_o', 380), ('car', 343988), ('car_o', 11822), ('motorcyclist', 31352), ('motorcyclist_o', 4376), ('neg', 9288), ('truck', 45749), ('truck_o', 497)]
    # 数据集: allinone_new, [('bus', 1813), ('bus_o', 36), ('car', 25756), ('car_o', 1159), ('motorcyclist', 1590), ('motorcyclist_o', 173), ('neg', 690), ('truck', 2870), ('truck_o', 39)]

    # RM_ADAS_AllInOne
    # 类别: "car", "bus", "truck", "motorcyclist", "plate", "fuzzy_plate", "moto_plate", "mote_fuzzy_plate"
    # 注: motorcycle 表示没人骑行的数据，这里不参与训练
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate/"
    # args.select_name_list = ["car", "bus", "truck", "motorcyclist", "plate", "fuzzy_plate", "moto_plate", "mote_fuzzy_plate"]
    # args.set_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "license_plate", "moto_license_plate", "moto_license_plate"]
    # args.finnal_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "moto_license_plate", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "XML/"
    # 数据集: allinone_w_licenseplate, [('bus', 4464), ('bus_o', 24), ('car', 43292), ('car_o', 847), ('license_plate', 16907), ('license_plate_ignore', 39), ('moto_license_plate', 147), ('motorcyclist', 5600), ('motorcyclist_o', 512), ('neg', 452), ('truck', 2535), ('truck_o', 17)]

    # RM_ADAS_AllInOne
    # 类别: "car", "bus", "truck", "motorcyclist", "licence", "licence_f"（汽车车牌）
    # 注: motorcycle 表示没人骑行的数据，这里不参与训练
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
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph_new_style/ADAS_AllInOne_New_Test/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph_new_style/ADAS_VehicleTailRegression/"
    # args.select_name_list = ["car", "bus", "truck", "motorcyclist", "licence", "licence_f"]
    # args.set_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "license_plate"]
    # args.finnal_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "Annotations/"
    # 数据集: ADAS_ADPLUS2.0, [('bus', 189), ('car', 2883), ('car_o', 2), ('license_plate', 1106), ('license_plate_ignore', 186), ('motorcyclist', 1080), ('motorcyclist_o', 24), ('neg', 101), ('truck', 371), ('truck_o', 1)]
    # 数据集: ADAS_ADPLUS2.0_5M_Backlight, [('bus', 309), ('car', 4127), ('car_o', 27), ('license_plate', 2228), ('license_plate_ignore', 820), ('motorcyclist', 1432), ('motorcyclist_o', 150), ('neg', 70), ('truck', 428)]
    # 数据集: ADAS_ADPLUS2.0_NearPerson, [('car', 2459), ('car_o', 194), ('license_plate', 877), ('license_plate_ignore', 786), ('motorcyclist', 69), ('motorcyclist_o', 148), ('neg', 2552), ('truck', 19), ('truck_o', 2)]
    # 数据集: ADAS_AllInOne_Backlight_AbnormalVehicle, [('bus', 404), ('bus_o', 2), ('car', 8730), ('car_o', 570), ('license_plate', 3661), ('license_plate_ignore', 3098), ('motorcyclist', 1114), ('motorcyclist_o', 377), ('neg', 244), ('truck', 1683), ('truck_o', 16)]
    # 数据集: ADAS_AllInOne_MS1, [('bus', 1286), ('car', 19852), ('car_o', 27), ('license_plate', 8420), ('license_plate_ignore', 1218), ('motorcyclist', 2610), ('motorcyclist_o', 100), ('neg', 160), ('truck', 2456)]
    # 数据集: ADAS_AllInOne_Rainy_Night, [('bus', 295), ('car', 11305), ('car_o', 89), ('license_plate', 6163), ('license_plate_ignore', 2859), ('motorcyclist', 591), ('motorcyclist_o', 67), ('neg', 222), ('truck', 715)]
    # 数据集: ADAS_Night_Highway_Backlight, [('bus', 3), ('car', 1929), ('car_o', 244), ('license_plate', 1280), ('license_plate_ignore', 1261), ('neg', 130), ('truck', 3693), ('truck_o', 37)]
    # 数据集: ADAS_AbnormalVehicle, [('bus', 784), ('bus_o', 3), ('car', 46433), ('car_o', 16139), ('license_plate', 12454), ('license_plate_ignore', 9888), ('motorcyclist', 713), ('motorcyclist_o', 40), ('neg', 160), ('truck', 23543), ('truck_o', 1513)]
    # 数据集: ADAS_AllInOne_AbnormalVehicle, [('bus', 752), ('car', 9663), ('car_o', 323), ('license_plate', 5959), ('license_plate_ignore', 1574), ('motorcyclist', 383), ('motorcyclist_o', 21), ('neg', 1), ('truck', 4624), ('truck_o', 2)]
    # 数据集: ADAS_AllInOne_MS3_patch, [('bus', 1370), ('bus_o', 8), ('car', 16377), ('car_o', 263), ('license_plate', 7602), ('license_plate_ignore', 2880), ('motorcyclist', 1946), ('motorcyclist_o', 218), ('neg', 91), ('truck', 1994), ('truck_o', 4)] 

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
    # 数据集: england, [('car_bus_truck', 149602), ('car_bus_truck_o', 7354), ('license_plate', 56448), ('license_plate_ignore', 16461), ('neg', 455)]
    # 数据集: england_1080p, [('car_bus_truck', 134661), ('car_bus_truck_o', 28176), ('license_plate', 27280), ('license_plate_ignore', 24035), ('neg', 408)]
    # 数据集: france, [('car_bus_truck', 156347), ('car_bus_truck_o', 12089), ('license_plate', 59833), ('license_plate_ignore', 18644), ('neg', 965)]
    # 数据集: italy, [('car_bus_truck', 21393), ('car_bus_truck_o', 1080), ('license_plate', 8985), ('license_plate_ignore', 1830), ('neg', 66)]
    # 数据集: netherlands, [('car_bus_truck', 128340), ('car_bus_truck_o', 17745), ('license_plate', 39927), ('license_plate_ignore', 17321), ('neg', 887)]
    # 数据集: moni, [('car_bus_truck', 23613), ('car_bus_truck_o', 18), ('license_plate', 13516), ('license_plate_ignore', 753)]
    # 数据集: moni_0415, [('car_bus_truck', 7445), ('car_bus_truck_o', 1), ('license_plate', 5021), ('license_plate_ignore', 576), ('neg', 8439)]
    # 数据集: hardNeg, [('car_bus_truck', 54352), ('car_bus_truck_o', 35398), ('license_plate', 11068), ('license_plate_ignore', 17030), ('neg', 200)]

    # # ZG_ZHJYZ_detection
    # # 注：shaobing 无车牌标注
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/jiayouzhan/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/jiayouzhan_5M/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/sandaofangxian/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/shenzhentiaoqiao/"
    # # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/anhuihuaibeigaosu/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/anhuihuaibeigaosu_night_diguangzhao/"
    # # ignore args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/shaobing/"
    # args.select_name_list = [ "plate", "fuzzy_plate", "roi_ignore_plate"]
    # args.set_name_list = [ "license", "license", "license" ]
    # args.finnal_name_list = [ "license", "neg"]
    # args.select_name_list = ["car", "bus", "truck", "motorcyclist", "plate", "fuzzy_plate"]
    # args.set_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "license_plate"]
    # args.finnal_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "XML/"
    # 数据集: jiayouzhan, [('bus', 682), ('bus_o', 3), ('car', 12232), ('car_o', 63), ('license_plate', 11374), ('license_plate_ignore', 218), ('motorcyclist', 44), ('truck', 180)]
    # 数据集: jiayouzhan_5M, [('bus', 488), ('bus_o', 1), ('car', 7769), ('car_o', 5), ('license_plate', 5172), ('motorcyclist', 52), ('truck', 22)]
    # 数据集: sandaofangxian, [('bus', 1776), ('bus_o', 8), ('car', 8686), ('car_o', 540), ('license_plate', 9949), ('license_plate_ignore', 6), ('truck', 13053), ('truck_o', 214)]
    # 数据集: shenzhentiaoqiao, [('bus', 5446), ('bus_o', 4), ('car', 29330), ('car_o', 116), ('license_plate', 21606), ('license_plate_ignore', 2640), ('motorcyclist', 339), ('motorcyclist_o', 4), ('truck', 776), ('truck_o', 5)]
    # 数据集: anhuihuaibeigaosu, [('bus', 2783), ('car', 231027), ('car_o', 832), ('license_plate', 95038), ('license_plate_ignore', 13417), ('truck', 46598), ('truck_o', 82)]
    # 数据集: anhuihuaibeigaosu_night_diguangzhao, [('bus', 3629), ('car', 80326), ('car_o', 13), ('license_plate', 28492), ('license_plate_ignore', 1618), ('truck', 11479), ('truck_o', 1)]

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
    # 数据集: daminghu, [('bus', 2910), ('bus_o', 1), ('car', 26480), ('car_o', 29), ('motorcyclist', 18449), ('motorcyclist_o', 22), ('neg', 192), ('truck', 193)]
    # 数据集: daminghu_night, [('bus', 158), ('bus_o', 1), ('car', 3220), ('motorcyclist', 3823), ('motorcyclist_o', 4), ('neg', 161), ('truck', 18)]
    # 数据集: shandongyingzikou, [('bus', 124), ('bus_o', 1), ('car', 8090), ('car_o', 271), ('motorcyclist', 361), ('motorcyclist_o', 81), ('truck', 4056), ('truck_o', 11)]
    # 数据集: shandongyingzikou_night_hongwai, [('bus', 29), ('car', 1894), ('car_o', 103), ('motorcyclist', 121), ('motorcyclist_o', 28), ('truck', 851), ('truck_o', 1)]
    # 数据集: yongzou_night_hongwai, [('bus', 82), ('car', 2895), ('car_o', 1), ('motorcyclist', 12989), ('motorcyclist_o', 69), ('neg', 10), ('truck', 167)]
    # 数据集: shenzhenlukou, [('bus', 637), ('bus_o', 8), ('car', 14514), ('car_o', 713), ('motorcyclist', 362), ('motorcyclist_o', 46), ('truck', 373)]
    # 数据集: shenzhenlukou_night_hongwai, [('bus', 604), ('car', 13723), ('car_o', 21), ('motorcyclist', 1312), ('motorcyclist_o', 8), ('neg', 13), ('truck', 291)]
    # 数据集: shenzhenlukou_night_diguangzhao, [('bus', 148), ('car', 10593), ('car_o', 124), ('motorcyclist', 392), ('motorcyclist_o', 13), ('neg', 22), ('truck', 269)]
    # 数据集: rongheng, [('bus', 1653), ('bus_o', 10), ('car', 39129), ('car_o', 1039), ('motorcyclist', 5368), ('motorcyclist_o', 209), ('neg', 1913), ('truck', 674), ('truck_o', 1)]
    # 数据集: [('bus', 714), ('car', 36126), ('car_o', 16), ('motorcyclist', 3829), ('motorcyclist_o', 155), ('neg', 1465), ('truck', 343)], 

    # RM_C27_detection
    # 类别: "car", "license_plate"（汽车车牌）
    # 注：license_plate 清晰车牌，不可用
    # args.input_dir = "/yuanhuan/data/image/RM_C27_detection/zd_c27_2020_0209_1125/"
    # args.select_name_list = ["car", "license_plate"]
    # args.set_name_list = ["car_bus_truck", "license_plate"]
    # args.finnal_name_list = ["car_bus_truck", "license_plate", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "XML/"
    
    # # RM_C27_detection
    # # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil/Brazil_all/"
    # args.input_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_new_style/shate_20230308/"
    # args.select_name_list = ["car", "bus", "truck", "motorcycle", "motorcyclist", "lince-plate", "fuzzy-plate", "lince-motorplate", "fuzzy-motorplate", "license"]
    # args.set_name_list = ["car", "bus", "truck", "motorcyclist", "motorcyclist", "license_plate", "license_plate", "moto_license_plate", "moto_license_plate", "license_plate"]
    # args.finnal_name_list = ["car", "bus", "truck", "motorcyclist", "license_plate", "moto_license_plate", "neg"]
    # args.jpg_dir = args.input_dir + "JPEGImages/"  
    # args.xml_dir = args.input_dir + "Annotations/"
    # 数据集: Brazil_all, [('bus', 15059), ('bus_o', 106), ('car', 417109), ('car_o', 15929), ('license_plate', 197924), ('moto_license_plate', 41996), ('motorcyclist', 58165), ('motorcyclist_o', 506), ('neg', 7), ('truck', 34638), ('truck_o', 292)]
    # 数据集: shate_20230308, [('bus', 1200), ('bus_o', 5), ('car', 84808), ('car_o', 4602), ('license_plate', 40107), ('motorcyclist', 98), ('truck', 7057), ('truck_o', 25)]

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