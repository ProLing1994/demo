import argparse
import numpy as np
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET


def statistic_classname(args):
    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]
    
    classname_dict = dict()
    for idx in tqdm(range(len(jpg_list))):
        xml_path = os.path.join(args.xml_dir, jpg_list[idx].replace(".jpg", ".xml"))

        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

        for object in root.findall('object'):
            classname = str(object.find('name').text)
            
            if classname not in classname_dict:
                classname_dict[classname] = 1
            else:
                classname_dict[classname] += 1

    classname_dict = sorted(classname_dict.items(), key=lambda k: k[0])
    print(classname_dict)


def statistic_classname_train_val_test(args):
    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]

    classname_dict = dict()
    for statistic_key in args.statistic_dict.keys():

        filename = args.statistic_dict[statistic_key]
        with open(filename, "r") as f:
            for line in tqdm(f):
                xml_path = os.path.join(args.xml_dir, line.strip() + ".xml")

                tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
                root = tree.getroot()   # 获取根节点

                for object in root.findall('object'):
                    classname = str(object.find('name').text)
                    
                    if classname not in classname_dict:

                        classname_dict[classname] = dict()
                        for temp_key in args.statistic_dict.keys():
                            classname_dict[classname][temp_key] = 0

                        classname_dict[classname][statistic_key] = 1
                    else:
                        classname_dict[classname][statistic_key] += 1

    classname_dict = sorted(classname_dict.items(), key=lambda k: k[0])
    print(classname_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # 数据集：China，包含 car\person\person_o\license_plate，car 包含 bus\truck
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/China/"         
    # [(' upspeed_20', 2), (' upspeed_5', 1), ('bicycle', 8978), ('bicycle ', 1), ('car', 165480), ('downspeed_40', 3), ('downspeed_60', 46), ('downspeed_70', 37), ('height_2.2', 4), ('height_2.5', 6), ('height_2.6', 1), ('height_3', 1), ('height_3.8', 4), ('height_4', 2), ('height_4.0', 10), ('height_4.2', 4), ('height_4.5', 167), ('height_4.7', 5), ('height_4.8', 3), ('height_5', 140), ('height_5.0', 127), ('height_5.5', 49), ('license_plate', 35630), ('motorcycle', 8589), ('person', 40085), ('person_o', 9257), ('sign_stop', 303), ('traffic_sign', 1), ('trafficlight_green_l', 266), ('trafficlight_green_n', 1321), ('trafficlight_green_r', 110), ('trafficlight_green_u', 466), ('trafficlight_green_u ', 1), ('trafficlight_green_x', 541), ('trafficlight_off', 654), ('trafficlight_red_l', 803), ('trafficlight_red_l ', 1), ('trafficlight_red_n', 1537), ('trafficlight_red_n ', 3), ('trafficlight_red_r', 29), ('trafficlight_red_u', 467), ('trafficlight_red_x', 799), ('trafficlight_special', 42), ('trafficlight_yellow_l', 33), ('trafficlight_yellow_n', 221), ('trafficlight_yellow_r', 7), ('trafficlight_yellow_u', 33), ('trafficlight_yellow_x', 39), ('unspeed_30', 52), ('upspeed_10', 26), ('upspeed_15', 35), ('upspeed_20', 325), ('upspeed_30', 914), ('upspeed_40', 870), ('upspeed_5', 53), ('upspeed_50', 241), ('upspeed_60', 711), ('upspeed_70', 19), ('upspeed_80', 148), ('weight_15', 5), ('weight_20', 13), ('weight_30', 21), ('weight_40', 170), ('weight_49', 3), ('weight_50', 12), ('weight_55', 34), ('weightr_10', 2), ('weightr_13', 6), ('weightr_14', 8), ('zebra_crossing', 9718)
    # [(' upspeed_20', {'trainval': 2, 'test': 0}), (' upspeed_5', {'trainval': 1, 'test': 0}), ('bicycle', {'trainval': 8121, 'test': 857}), ('bicycle ', {'trainval': 1, 'test': 0}), ('car', {'trainval': 148905, 'test': 16575}), ('downspeed_40', {'trainval': 3, 'test': 0}), ('downspeed_60', {'trainval': 41, 'test': 5}), ('downspeed_70', {'trainval': 33, 'test': 4}), ('height_2.2', {'trainval': 3, 'test': 1}), ('height_2.5', {'trainval': 4, 'test': 2}), ('height_2.6', {'trainval': 1, 'test': 0}), ('height_3', {'trainval': 1, 'test': 0}), ('height_3.8', {'trainval': 3, 'test': 1}), ('height_4', {'trainval': 1, 'test': 1}), ('height_4.0', {'trainval': 9, 'test': 1}), ('height_4.2', {'trainval': 3, 'test': 1}), ('height_4.5', {'trainval': 151, 'test': 16}), ('height_4.7', {'trainval': 5, 'test': 0}), ('height_4.8', {'trainval': 3, 'test': 0}), ('height_5', {'trainval': 128, 'test': 12}), ('height_5.0', {'trainval': 117, 'test': 10}), ('height_5.5', {'trainval': 40, 'test': 9}), ('license_plate', {'trainval': 32104, 'test': 3526}), ('motorcycle', {'trainval': 7696, 'test': 893}), ('person', {'trainval': 35964, 'test': 4121}), ('person_o', {'trainval': 8296, 'test': 961}), ('sign_stop', {'trainval': 275, 'test': 28}), ('traffic_sign', {'trainval': 1, 'test': 0}), ('trafficlight_green_l', {'trainval': 237, 'test': 29}), ('trafficlight_green_n', {'trainval': 1199, 'test': 122}), ('trafficlight_green_r', {'trainval': 96, 'test': 14}), ('trafficlight_green_u', {'trainval': 422, 'test': 44}), ('trafficlight_green_u ', {'trainval': 0, 'test': 1}), ('trafficlight_green_x', {'trainval': 486, 'test': 55}), ('trafficlight_off', {'trainval': 575, 'test': 79}), ('trafficlight_red_l', {'trainval': 717, 'test': 86}), ('trafficlight_red_l ', {'trainval': 0, 'test': 1}), ('trafficlight_red_n', {'trainval': 1368, 'test': 169}), ('trafficlight_red_n ', {'trainval': 3, 'test': 0}), ('trafficlight_red_r', {'trainval': 26, 'test': 3}), ('trafficlight_red_u', {'trainval': 422, 'test': 45}), ('trafficlight_red_x', {'trainval': 717, 'test': 82}), ('trafficlight_special', {'trainval': 34, 'test': 8}), ('trafficlight_yellow_l', {'trainval': 27, 'test': 6}), ('trafficlight_yellow_n', {'trainval': 202, 'test': 19}), ('trafficlight_yellow_r', {'trainval': 7, 'test': 0}), ('trafficlight_yellow_u', {'trainval': 30, 'test': 3}), ('trafficlight_yellow_x', {'trainval': 35, 'test': 4}), ('unspeed_30', {'trainval': 46, 'test': 6}), ('upspeed_10', {'trainval': 25, 'test': 1}), ('upspeed_15', {'trainval': 32, 'test': 3}), ('upspeed_20', {'trainval': 293, 'test': 32}), ('upspeed_30', {'trainval': 830, 'test': 84}), ('upspeed_40', {'trainval': 785, 'test': 85}), ('upspeed_5', {'trainval': 48, 'test': 5}), ('upspeed_50', {'trainval': 219, 'test': 22}), ('upspeed_60', {'trainval': 649, 'test': 62}), ('upspeed_70', {'trainval': 17, 'test': 2}), ('upspeed_80', {'trainval': 130, 'test': 18}), ('weight_15', {'trainval': 4, 'test': 1}), ('weight_20', {'trainval': 12, 'test': 1}), ('weight_30', {'trainval': 20, 'test': 1}), ('weight_40', {'trainval': 150, 'test': 20}), ('weight_49', {'trainval': 3, 'test': 0}), ('weight_50', {'trainval': 10, 'test': 2}), ('weight_55', {'trainval': 33, 'test': 1}), ('weightr_10', {'trainval': 1, 'test': 1}), ('weightr_13', {'trainval': 5, 'test': 1}), ('weightr_14', {'trainval': 7, 'test': 1}), ('zebra_crossing', {'trainval': 8763, 'test': 955})]
    
    # 数据集：China_6mm，包含 car\license_plate，car 包含 bus\truck
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/China_6mm/"     
    # # {'car': 6206, 'license_plate': 2665}
    # [('car', {'trainval': 5603, 'test': 603}), ('license_plate', {'trainval': 2407, 'test': 258})]

    # 数据集：Europe，包含 license_plate
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/Europe/"        
    # # {'license_plate': 24836}
    # [('license_plate', {'trainval': 22339, 'test': 2497})]

    # 数据集：Mexico，包含 car\license_plate，car 包含 bus\truck
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/Mexico/"        
    # # {'car': 47367, 'license_plate': 17805}
    # [('car', {'trainval': 42615, 'test': 4752}), ('license_plate', {'trainval': 15997, 'test': 1808})]
    
    # 数据集：RM_ADAS_AllInOne，包含 car\bus\truck\person\person_o
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone/"  
    # # [('SEWER', 4), ('\\', 1), ('arrow', 1), ('arrow_L', 3), ('arrow_g', 773), ('arrow_gl', 6), ('arrow_gu', 2), ('arrow_l', 4272), ('arrow_lg', 178), ('arrow_lr', 275), ('arrow_lu', 1870), ('arrow_lur', 475), ('arrow_r', 6293), ('arrow_rg', 59), ('arrow_rl', 15), ('arrow_rlu', 11), ('arrow_ru', 6), ('arrow_u', 31109), ('arrow_ug', 146), ('arrow_ul', 1637), ('arrow_ulr', 20), ('arrow_ur', 6313), ('arrow_url', 81), ('bicycle', 14848), ('bicyclist', 19982), ('bus', 48437), ('car', 355810), ('car_big', 180), ('car_big_front', 14741), ('car_big_reg', 65668), ('car_front', 43826), ('car_reg', 261084), ('cat', 3), ('detour_r', 2), ('dog', 172), ('downhill_long', 8), ('downspeed_30', 3), ('downspeed_40', 12), ('downspeed_5', 1), ('downspeed_50', 11), ('downspeed_60', 274), ('downspeed_70', 205), ('height_1.9', 9), ('height_2', 64), ('height_2.2', 7), ('height_2.5', 6), ('height_2.6', 1), ('height_3', 8), ('height_3.2', 2), ('height_3.5', 12), ('height_3.8', 88), ('height_4', 25), ('height_4.1', 7), ('height_4.2', 79), ('height_4.5', 515), ('height_4.7', 38), ('height_5', 949), ('height_5.5', 138), ('height_6', 3), ('motorcycle', 3366), ('motorcyclist', 35728), ('person', 110442), ('person_o', 18976), ('preson', 8), ('sewer', 61184), ('sewer.', 2), ('sign_accident', 112), ('sign_nostop', 36), ('sign_nostopd', 50), ('sign_nostops', 6997), ('sign_stop', 1802), ('sign_stops', 25), ('spupeed_60', 3), ('traffic_red_n', 2), ('trafficlight_greeb_r', 5), ('trafficlight_greeb_x', 3), ('trafficlight_greed_n', 3), ('trafficlight_greed_x', 2), ('trafficlight_green_25', 2), ('trafficlight_green_4', 2), ('trafficlight_green_g', 14), ('trafficlight_green_l', 873), ('trafficlight_green_l_04', 2), ('trafficlight_green_l_1', 2), ('trafficlight_green_l_10', 2), ('trafficlight_green_l_11', 2), ('trafficlight_green_l_13', 2), ('trafficlight_green_l_17', 1), ('trafficlight_green_l_23', 1), ('trafficlight_green_l_3', 1), ('trafficlight_green_l_32', 1), ('trafficlight_green_l_5', 1), ('trafficlight_green_l_55', 1), ('trafficlight_green_l_56', 2), ('trafficlight_green_l_58', 1), ('trafficlight_green_l_59', 1), ('trafficlight_green_l_7', 1), ('trafficlight_green_l_76', 1), ('trafficlight_green_l_77', 1), ('trafficlight_green_l_9', 1), ('trafficlight_green_m', 31), ('trafficlight_green_ml', 4), ('trafficlight_green_mu', 4), ('trafficlight_green_n', 8917), ('trafficlight_green_n_04', 1), ('trafficlight_green_n_10', 1), ('trafficlight_green_n_12', 1), ('trafficlight_green_n_13', 3), ('trafficlight_green_n_14', 1), ('trafficlight_green_n_2', 2), ('trafficlight_green_n_20', 1), ('trafficlight_green_n_24', 1), ('trafficlight_green_n_25', 2), ('trafficlight_green_n_3', 1), ('trafficlight_green_n_4', 8), ('trafficlight_green_n_5', 16), ('trafficlight_green_n_6', 16), ('trafficlight_green_n_7', 15), ('trafficlight_green_n_8', 16), ('trafficlight_green_n_9', 8), ('trafficlight_green_p', 1680), ('trafficlight_green_p_06', 1), ('trafficlight_green_p_10', 1), ('trafficlight_green_p_12', 1), ('trafficlight_green_p_14', 1), ('trafficlight_green_p_25', 1), ('trafficlight_green_p_28', 2), ('trafficlight_green_p_29', 3), ('trafficlight_green_p_30', 1), ('trafficlight_green_p_31', 3), ('trafficlight_green_p_32', 2), ('trafficlight_green_p_36', 3), ('trafficlight_green_p_5', 1), ('trafficlight_green_p_52', 1), ('trafficlight_green_p_54', 1), ('trafficlight_green_p_55', 3), ('trafficlight_green_p_6', 1), ('trafficlight_green_p_7', 3), ('trafficlight_green_pn', 11), ('trafficlight_green_px', 24), ('trafficlight_green_r', 469), ('trafficlight_green_r_12', 1), ('trafficlight_green_r_13', 4), ('trafficlight_green_r_15', 2), ('trafficlight_green_r_17', 1), ('trafficlight_green_r_22', 1), ('trafficlight_green_r_26', 2), ('trafficlight_green_r_27', 1), ('trafficlight_green_r_28', 1), ('trafficlight_green_r_30', 2), ('trafficlight_green_r_31', 2), ('trafficlight_green_r_32', 4), ('trafficlight_green_r_33', 1), ('trafficlight_green_r_35', 1), ('trafficlight_green_r_4', 1), ('trafficlight_green_r_43', 1), ('trafficlight_green_r_45', 1), ('trafficlight_green_r_5', 1), ('trafficlight_green_r_55', 1), ('trafficlight_green_r_56', 2), ('trafficlight_green_r_58', 1), ('trafficlight_green_r_59', 2), ('trafficlight_green_u', 2031), ('trafficlight_green_u_11', 1), ('trafficlight_green_u_13', 1), ('trafficlight_green_u_15', 2), ('trafficlight_green_u_17', 2), ('trafficlight_green_u_20', 1), ('trafficlight_green_u_22', 2), ('trafficlight_green_u_23', 4), ('trafficlight_green_u_24', 4), ('trafficlight_green_u_25', 2), ('trafficlight_green_u_26', 5), ('trafficlight_green_u_27', 1), ('trafficlight_green_u_28', 2), ('trafficlight_green_u_29', 2), ('trafficlight_green_u_30', 1), ('trafficlight_green_u_32', 2), ('trafficlight_green_u_33', 7), ('trafficlight_green_u_34', 1), ('trafficlight_green_u_35', 1), ('trafficlight_green_u_43', 1), ('trafficlight_green_u_45', 4), ('trafficlight_green_u_47', 3), ('trafficlight_green_u_5', 2), ('trafficlight_green_u_52', 3), ('trafficlight_green_u_53', 2), ('trafficlight_green_u_55', 1), ('trafficlight_green_u_56', 2), ('trafficlight_green_u_58', 5), ('trafficlight_green_u_59', 2), ('trafficlight_green_u_67', 2), ('trafficlight_green_u_68', 1), ('trafficlight_green_u_8', 1), ('trafficlight_green_x', 3071), ('trafficlight_green_x_1', 2), ('trafficlight_green_x_11', 2), ('trafficlight_green_x_14', 2), ('trafficlight_green_x_15', 2), ('trafficlight_green_x_2', 1), ('trafficlight_green_x_20', 1), ('trafficlight_green_x_3', 1), ('trafficlight_green_x_5', 2), ('trafficlight_green_x_55', 2), ('trafficlight_green_x_61', 2), ('trafficlight_green_x_7', 1), ('trafficlight_green_x_8', 1), ('trafficlight_green_x_9', 2), ('trafficlight_grenn_n', 3), ('trafficlight_grenn_u_47', 1), ('trafficlight_off', 3223), ('trafficlight_off_n', 31), ('trafficlight_red', 1), ('trafficlight_red_12', 1), ('trafficlight_red_25_b', 2), ('trafficlight_red_26', 2), ('trafficlight_red_26_b', 1), ('trafficlight_red_27_b', 1), ('trafficlight_red_28_b', 1), ('trafficlight_red_3', 2), ('trafficlight_red_37_b', 4), ('trafficlight_red_38_b', 2), ('trafficlight_red_4', 4), ('trafficlight_red_5', 5), ('trafficlight_red_g', 77), ('trafficlight_red_g_14', 2), ('trafficlight_red_g_16', 1), ('trafficlight_red_g_17', 2), ('trafficlight_red_l', 3622), ('trafficlight_red_l_07', 1), ('trafficlight_red_l_10', 1), ('trafficlight_red_l_11', 1), ('trafficlight_red_l_12', 1), ('trafficlight_red_l_13', 3), ('trafficlight_red_l_15', 3), ('trafficlight_red_l_16', 1), ('trafficlight_red_l_17', 2), ('trafficlight_red_l_18', 3), ('trafficlight_red_l_19', 4), ('trafficlight_red_l_20', 3), ('trafficlight_red_l_21', 6), ('trafficlight_red_l_22', 2), ('trafficlight_red_l_23', 4), ('trafficlight_red_l_26', 2), ('trafficlight_red_l_28', 1), ('trafficlight_red_l_30', 2), ('trafficlight_red_l_31', 3), ('trafficlight_red_l_32', 2), ('trafficlight_red_l_33', 1), ('trafficlight_red_l_4', 1), ('trafficlight_red_l_44', 1), ('trafficlight_red_l_5', 1), ('trafficlight_red_l_6', 12), ('trafficlight_red_l_63', 2), ('trafficlight_red_l_65', 2), ('trafficlight_red_l_66', 1), ('trafficlight_red_l_67', 2), ('trafficlight_red_l_68', 1), ('trafficlight_red_l_7', 6), ('trafficlight_red_l_8', 2), ('trafficlight_red_l_84', 1), ('trafficlight_red_l_9', 4), ('trafficlight_red_m', 23), ('trafficlight_red_n', 17738), ('trafficlight_red_n\t 35', 1), ('trafficlight_red_n.', 1), ('trafficlight_red_n_01', 1), ('trafficlight_red_n_10', 1), ('trafficlight_red_n_11', 3), ('trafficlight_red_n_12', 3), ('trafficlight_red_n_13', 1), ('trafficlight_red_n_17', 3), ('trafficlight_red_n_18', 2), ('trafficlight_red_n_19', 9), ('trafficlight_red_n_2', 1), ('trafficlight_red_n_20', 5), ('trafficlight_red_n_21', 5), ('trafficlight_red_n_22', 2), ('trafficlight_red_n_3', 1), ('trafficlight_red_n_35', 1), ('trafficlight_red_n_38', 1), ('trafficlight_red_n_4', 1), ('trafficlight_red_n_45', 1), ('trafficlight_red_n_48', 1), ('trafficlight_red_n_7', 1), ('trafficlight_red_n_76', 1), ('trafficlight_red_n_8', 2), ('trafficlight_red_p', 2852), ('trafficlight_red_p\t06', 1), ('trafficlight_red_p 40', 1), ('trafficlight_red_p_12', 2), ('trafficlight_red_p_13', 1), ('trafficlight_red_p_14', 5), ('trafficlight_red_p_15', 1), ('trafficlight_red_p_20', 3), ('trafficlight_red_p_22', 1), ('trafficlight_red_p_23', 4), ('trafficlight_red_p_24', 4), ('trafficlight_red_p_25', 1), ('trafficlight_red_p_33', 4), ('trafficlight_red_p_37', 2), ('trafficlight_red_p_39', 1), ('trafficlight_red_p_41', 2), ('trafficlight_red_p_43', 1), ('trafficlight_red_p_44', 1), ('trafficlight_red_p_53', 1), ('trafficlight_red_p_57', 2), ('trafficlight_red_p_6', 5), ('trafficlight_red_p_7', 1), ('trafficlight_red_pn', 14), ('trafficlight_red_px', 45), ('trafficlight_red_px_10', 1), ('trafficlight_red_r', 560), ('trafficlight_red_r_05', 1), ('trafficlight_red_r_13', 5), ('trafficlight_red_r_132', 2), ('trafficlight_red_r_136', 2), ('trafficlight_red_r_137', 5), ('trafficlight_red_r_138', 2), ('trafficlight_red_r_15', 1), ('trafficlight_red_r_17', 1), ('trafficlight_red_r_18', 3), ('trafficlight_red_r_19', 2), ('trafficlight_red_r_24', 1), ('trafficlight_red_r_25', 2), ('trafficlight_red_r_3', 1), ('trafficlight_red_r_4', 7), ('trafficlight_red_r_5', 1), ('trafficlight_red_r_52', 3), ('trafficlight_red_r_53', 4), ('trafficlight_red_r_54', 3), ('trafficlight_red_r_57', 4), ('trafficlight_red_r_58', 3), ('trafficlight_red_u', 3061), ('trafficlight_red_u_01', 1), ('trafficlight_red_u_02', 1), ('trafficlight_red_u_03', 1), ('trafficlight_red_u_05', 2), ('trafficlight_red_u_102', 2), ('trafficlight_red_u_103', 2), ('trafficlight_red_u_104', 5), ('trafficlight_red_u_105', 2), ('trafficlight_red_u_12', 1), ('trafficlight_red_u_13', 4), ('trafficlight_red_u_15', 2), ('trafficlight_red_u_16', 1), ('trafficlight_red_u_17', 2), ('trafficlight_red_u_18', 1), ('trafficlight_red_u_19', 4), ('trafficlight_red_u_21', 4), ('trafficlight_red_u_22', 2), ('trafficlight_red_u_23', 6), ('trafficlight_red_u_24', 7), ('trafficlight_red_u_26', 1), ('trafficlight_red_u_27', 1), ('trafficlight_red_u_30', 1), ('trafficlight_red_u_31', 2), ('trafficlight_red_u_32', 2), ('trafficlight_red_u_33', 1), ('trafficlight_red_u_5', 1), ('trafficlight_red_u_6', 1), ('trafficlight_red_u_63', 1), ('trafficlight_red_u_7', 1), ('trafficlight_red_u_76', 2), ('trafficlight_red_u_77', 1), ('trafficlight_red_u_8', 5), ('trafficlight_red_u_9', 2), ('trafficlight_red_x', 6758), ('trafficlight_red_x_12', 1), ('trafficlight_red_x_14', 3), ('trafficlight_red_x_15', 3), ('trafficlight_red_x_20', 1), ('trafficlight_red_x_22', 3), ('trafficlight_red_x_23', 4), ('trafficlight_red_x_24', 1), ('trafficlight_red_x_29', 4), ('trafficlight_red_x_30', 1), ('trafficlight_red_x_32', 1), ('trafficlight_red_x_33', 1), ('trafficlight_red_x_38', 2), ('trafficlight_red_x_4', 1), ('trafficlight_red_x_52', 1), ('trafficlight_red_x_53', 2), ('trafficlight_red_x_57', 2), ('trafficlight_red_x_72', 2), ('trafficlight_red_x_77', 1), ('trafficlight_red_x_81', 1), ('trafficlight_rede_r_53', 1), ('trafficlight_ren_n', 1), ('trafficlight_ren_p', 1), ('trafficlight_special', 292), ('trafficlight_special_red_n_2', 1), ('trafficlight_yellow_l', 132), ('trafficlight_yellow_n', 862), ('trafficlight_yellow_p', 1), ('trafficlight_yellow_r', 51), ('trafficlight_yellow_u', 87), ('trafficlight_yellow_x', 167), ('trafficlight_yellow_x_5', 1), ('trafficright_green_l_5', 1), ('trafficright_green_p_15', 2), ('trafficright_green_p_20', 1), ('trafficright_green_p_25', 3), ('trafficright_green_p_46', 1), ('trafficright_green_p_52', 1), ('trafficright_green_p_53', 1), ('trafficright_green_p_9', 1), ('trafficright_green_u_15', 1), ('trafficright_green_u_20', 1), ('trafficright_green_u_21', 1), ('trafficright_green_u_22', 1), ('trafficright_green_u_7', 2), ('trafficright_red_l_23', 1), ('trafficright_red_p_12', 3), ('trafficright_red_p_14', 2), ('trafficright_red_p_16', 1), ('trafficright_red_p_2', 1), ('trafficright_red_p_23', 1), ('trafficright_red_p_26', 2), ('trafficright_red_p_33', 1), ('trafficright_red_p_6', 1), ('trafficright_red_p_8', 1), ('trafficright_red_r_11', 1), ('trafficright_red_u_23', 1), ('truck', 46246), ('turn_continue', 3), ('turn_l', 85), ('turn_r', 31), ('unspeed_10', 1), ('unspeed_20', 4), ('unspeed_30', 361), ('unspeed_40', 39), ('unspeed_5', 4), ('unspeed_50', 7), ('unspeed_60', 25), ('unspeed_70', 2), ('uphill_long', 2), ('upspee_30', 1), ('upspeed_10', 372), ('upspeed_100', 17), ('upspeed_120', 7), ('upspeed_15', 239), ('upspeed_20', 1677), ('upspeed_30', 3748), ('upspeed_40', 3557), ('upspeed_5', 212), ('upspeed_50', 864), ('upspeed_55', 1), ('upspeed_60', 2787), ('upspeed_70', 38), ('upspeed_80', 631), ('upspreed_60', 1), ('weight_10', 1), ('weight_13', 1), ('weight_15', 25), ('weight_20', 33), ('weight_25', 2), ('weight_30', 48), ('weight_40', 398), ('weight_49', 4), ('weight_5', 1), ('weight_50', 34), ('weight_55', 156), ('weightr_10', 4), ('weightr_13', 18), ('weightr_14', 24), ('weightr_2.5', 3), ('weightr_40', 14), ('weightr_50', 1), ('weightr_55', 1), ('zebra_crossing', 36924)]

    # 数据集：RM_ADAS_AllInOne allinone_lincense_plate，包含 car\bus\truck\plate\fuzzy_plate\painted_plate\moto_plate\mote_fuzzy_plate
    # args.input_dir = "/mnt/huanyuan2/data/image/RM_ADAS_AllInOne/allinone_lincense_plate/"  
    # [('arrow', 1), ('arrow_g', 11), ('arrow_l', 18), ('arrow_lg', 8), ('arrow_lr', 2), ('arrow_lu', 9), ('arrow_lur', 14), ('arrow_r', 19), ('arrow_u', 248), ('arrow_ug', 4), ('arrow_ul', 3), ('arrow_ur', 50), ('bicycle', 179), ('bicyclist', 329), ('bus', 1486), ('car', 2798), ('car_big_front', 58), ('car_big_reg', 1519), ('car_front', 323), ('car_reg', 1926), ('fuzzy_plate', 542), ('height_2', 16), ('height_3.5', 4), ('height_5', 20), ('motorcycle', 23), ('motorcyclist', 494), ('painted_plate', 496), ('person', 1639), ('person_o', 300), ('plate', 1270), ('preson', 8), ('sewer', 321), ('sign_accident', 6), ('sign_nostops', 48), ('sign_stop', 3), ('trafficlight_green_l', 5), ('trafficlight_green_n', 128), ('trafficlight_green_p', 75), ('trafficlight_green_p_28', 2), ('trafficlight_green_p_29', 3), ('trafficlight_green_p_30', 1), ('trafficlight_green_p_31', 3), ('trafficlight_green_p_32', 2), ('trafficlight_green_r', 22), ('trafficlight_green_u', 22), ('trafficlight_off', 21), ('trafficlight_red_l', 5), ('trafficlight_red_n', 356), ('trafficlight_red_p', 87), ('trafficlight_red_r', 28), ('trafficlight_red_x', 7), ('trafficlight_yellow_l', 3), ('trafficlight_yellow_n', 8), ('trafficright_green_p_9', 1), ('truck', 215), ('unspeed_30', 1), ('unspeed_40', 9), ('upspeed_10', 6), ('upspeed_15', 16), ('upspeed_20', 17), ('upspeed_30', 37), ('upspeed_40', 8), ('weight_40', 3), ('zebra_crossing', 789)]

    # 数据集：ZG_ZHJYZ_detection，包含 car\bus\truck\plate\fuzzy_plate
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan/"
    # [('bus', 685), ('car', 12295), ('fuzzy_plate', 1792), ('plate', 9800), ('roi_ignore_plate', 227), ('truck', 180)]
    # [('bus', {'trainval': 607, 'test': 78}), ('car', {'trainval': 11027, 'test': 1268}), ('fuzzy_plate', {'trainval': 1589, 'test': 203}), ('plate', {'trainval': 8806, 'test': 994}), ('roi_ignore_plate', {'trainval': 203, 'test': 24}), ('truck', {'trainval': 160, 'test': 20})]
    # args.input_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_5M/"
    # [('bus', 488), ('car', 7776), ('fuzzy_plate', 1158), ('painted_plate', 175), ('plate', 4014), ('truck', 22)]
    # [('bus', {'trainval': 444, 'test': 44}), ('car', {'trainval': 7002, 'test': 774}), ('fuzzy_plate', {'trainval': 1030, 'test': 128}), ('painted_plate', {'trainval': 161, 'test': 14}), ('plate', {'trainval': 3628, 'test': 386}), ('truck', {'trainval': 20, 'test': 2})]
    # args.input_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/sandaofangxian/"
    # [('bus', 1784), ('car', 8992), ('fuzzy_plate', 3799), ('painted_plate', 394), ('plate', 5297), ('truck', 12537)]
    # [('bus', {'trainval': 1632, 'test': 152}), ('car', {'trainval': 8096, 'test': 896}), ('fuzzy_plate', {'trainval': 3403, 'test': 396}), ('painted_plate', {'trainval': 349, 'test': 45}), ('plate', {'trainval': 4765, 'test': 532}), ('truck', {'trainval': 11271, 'test': 1266})]

    # ZG_AHHBGS_detection car\bus\truck\plate\fuzzy_plate
    # args.input_dir = "/yuanhuan/data/image/ZG_AHHBGS_detection/anhuihuaibeigaosu/"
    # [('bus', 1406), ('car', 105548), ('fuzzy_plate', 30659), ('planted_plate', 363), ('plate', 7888), ('truck', 16424)]
    # [('bus', {'trainval': 1258, 'test': 148}), ('car', {'trainval': 94584, 'test': 10964}), ('fuzzy_plate', {'trainval': 27440, 'test': 3219}), ('planted_plate', {'trainval': 329, 'test': 34}), ('plate', {'trainval': 7122, 'test': 766}), ('truck', {'trainval': 14777, 'test': 1647})]

    # 数据集：ZG_ZHJYZ_detection 加油站测试样本
    args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/"
    # 2MB: [('car', 155), ('fuzzy_plate', 37), ('plate', 87), ('truck', 1)]
    # 2MH: [('bus', 7), ('car', 570), ('fuzzy_plate', 27), ('plate', 411), ('truck', 9)]
    # 5MB: [('bus', 13), ('car', 453), ('fuzzy_plate', 103), ('plate', 235), ('truck', 8)]
    # 5MH: [('bus', 18), ('car', 717), ('fuzzy_plate', 26), ('plate', 384), ('truck', 3)]
    # AHHBAS_418: [('car', 193), ('plate', 53), ('truck', 6)]
    # AHHBAS_43c: [('car', 180), ('fuzzy_plate', 25), ('plate', 58), ('truck', 5)]
    # AHHBAS_41a: [('bus', 5), ('car', 245), ('fuzzy_plate', 30), ('plate', 151), ('truck', 77)]
    # AHHBAS_41c: [('bus', 5), ('car', 97), ('fuzzy_plate', 58), ('plate', 38), ('truck', 93)]
    # SDFX_B1: [('car', 258), ('fuzzy_plate', 110), ('painted_plate', 50), ('plate', 50), ('truck', 362)]
    # SDFX_B2: [('bus', 6), ('car', 287), ('fuzzy_plate', 148), ('painted_plate', 53), ('plate', 68), ('truck', 360)]
    # SDFX_H1: [('bus', 3), ('car', 263), ('fuzzy_plate', 108), ('painted_plate', 27), ('plate', 55), ('truck', 367)]
    # SDFX_H2: [('bus', 1), ('car', 257), ('fuzzy_plate', 95), ('painted_plate', 28), ('plate', 76), ('truck', 314)]

    args.trainval_file = args.input_dir + "ImageSets/Main/trainval.txt"
    args.train_file = args.input_dir + "ImageSets/Main/train.txt"
    args.val_file = args.input_dir + "ImageSets/Main/val.txt"
    args.test_file = args.input_dir + "ImageSets/Main/test.txt"
    args.statistic_dict = {'trainval': args.trainval_file, 'test': args.test_file }

    # args.jpg_dir =  args.input_dir + "JPEGImages/"
    # args.xml_dir =  args.input_dir + "XML/"

    args.jpg_dir =  args.input_dir + "SDFX_H2/"
    args.xml_dir =  args.input_dir + "SDFX_H2_XML/"

    statistic_classname(args)
    # statistic_classname_train_val_test(args)