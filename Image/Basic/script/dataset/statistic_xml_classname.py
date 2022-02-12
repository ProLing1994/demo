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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # 数据集：China，包含 car\person\person_o\license_plate，car 包含 bus\truck
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/China/"         # [(' upspeed_20', 2), (' upspeed_5', 1), ('bicycle', 8978), ('bicycle ', 1), ('car', 165480), ('downspeed_40', 3), ('downspeed_60', 46), ('downspeed_70', 37), ('height_2.2', 4), ('height_2.5', 6), ('height_2.6', 1), ('height_3', 1), ('height_3.8', 4), ('height_4', 2), ('height_4.0', 10), ('height_4.2', 4), ('height_4.5', 167), ('height_4.7', 5), ('height_4.8', 3), ('height_5', 140), ('height_5.0', 127), ('height_5.5', 49), ('license_plate', 35630), ('motorcycle', 8589), ('person', 40085), ('person_o', 9257), ('sign_stop', 303), ('traffic_sign', 1), ('trafficlight_green_l', 266), ('trafficlight_green_n', 1321), ('trafficlight_green_r', 110), ('trafficlight_green_u', 466), ('trafficlight_green_u ', 1), ('trafficlight_green_x', 541), ('trafficlight_off', 654), ('trafficlight_red_l', 803), ('trafficlight_red_l ', 1), ('trafficlight_red_n', 1537), ('trafficlight_red_n ', 3), ('trafficlight_red_r', 29), ('trafficlight_red_u', 467), ('trafficlight_red_x', 799), ('trafficlight_special', 42), ('trafficlight_yellow_l', 33), ('trafficlight_yellow_n', 221), ('trafficlight_yellow_r', 7), ('trafficlight_yellow_u', 33), ('trafficlight_yellow_x', 39), ('unspeed_30', 52), ('upspeed_10', 26), ('upspeed_15', 35), ('upspeed_20', 325), ('upspeed_30', 914), ('upspeed_40', 870), ('upspeed_5', 53), ('upspeed_50', 241), ('upspeed_60', 711), ('upspeed_70', 19), ('upspeed_80', 148), ('weight_15', 5), ('weight_20', 13), ('weight_30', 21), ('weight_40', 170), ('weight_49', 3), ('weight_50', 12), ('weight_55', 34), ('weightr_10', 2), ('weightr_13', 6), ('weightr_14', 8), ('zebra_crossing', 9718)
    
    # 数据集：China_6mm，包含 car\license_plate，car 包含 bus\truck
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/China_6mm/"     # {'car': 6206, 'license_plate': 2665}

    # 数据集：Europe，包含 license_plate
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/Europe/"        # {'license_plate': 24836}

    # 数据集：Mexico，包含 car\license_plate，car 包含 bus\truck
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/Mexico/"        # {'car': 47367, 'license_plate': 17805}

    # 数据集：RM_ADAS_AllInOne，包含 car\bus\truck\person\person_o
    args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone/"  # [('SEWER', 4), ('\\', 1), ('arrow', 1), ('arrow_L', 3), ('arrow_g', 773), ('arrow_gl', 6), ('arrow_gu', 2), ('arrow_l', 4272), ('arrow_lg', 178), ('arrow_lr', 275), ('arrow_lu', 1870), ('arrow_lur', 475), ('arrow_r', 6293), ('arrow_rg', 59), ('arrow_rl', 15), ('arrow_rlu', 11), ('arrow_ru', 6), ('arrow_u', 31109), ('arrow_ug', 146), ('arrow_ul', 1637), ('arrow_ulr', 20), ('arrow_ur', 6313), ('arrow_url', 81), ('bicycle', 14848), ('bicyclist', 19982), ('bus', 48437), ('car', 355810), ('car_big', 180), ('car_big_front', 14741), ('car_big_reg', 65668), ('car_front', 43826), ('car_reg', 261084), ('cat', 3), ('detour_r', 2), ('dog', 172), ('downhill_long', 8), ('downspeed_30', 3), ('downspeed_40', 12), ('downspeed_5', 1), ('downspeed_50', 11), ('downspeed_60', 274), ('downspeed_70', 205), ('height_1.9', 9), ('height_2', 64), ('height_2.2', 7), ('height_2.5', 6), ('height_2.6', 1), ('height_3', 8), ('height_3.2', 2), ('height_3.5', 12), ('height_3.8', 88), ('height_4', 25), ('height_4.1', 7), ('height_4.2', 79), ('height_4.5', 515), ('height_4.7', 38), ('height_5', 949), ('height_5.5', 138), ('height_6', 3), ('motorcycle', 3366), ('motorcyclist', 35728), ('person', 110442), ('person_o', 18976), ('preson', 8), ('sewer', 61184), ('sewer.', 2), ('sign_accident', 112), ('sign_nostop', 36), ('sign_nostopd', 50), ('sign_nostops', 6997), ('sign_stop', 1802), ('sign_stops', 25), ('spupeed_60', 3), ('traffic_red_n', 2), ('trafficlight_greeb_r', 5), ('trafficlight_greeb_x', 3), ('trafficlight_greed_n', 3), ('trafficlight_greed_x', 2), ('trafficlight_green_25', 2), ('trafficlight_green_4', 2), ('trafficlight_green_g', 14), ('trafficlight_green_l', 873), ('trafficlight_green_l_04', 2), ('trafficlight_green_l_1', 2), ('trafficlight_green_l_10', 2), ('trafficlight_green_l_11', 2), ('trafficlight_green_l_13', 2), ('trafficlight_green_l_17', 1), ('trafficlight_green_l_23', 1), ('trafficlight_green_l_3', 1), ('trafficlight_green_l_32', 1), ('trafficlight_green_l_5', 1), ('trafficlight_green_l_55', 1), ('trafficlight_green_l_56', 2), ('trafficlight_green_l_58', 1), ('trafficlight_green_l_59', 1), ('trafficlight_green_l_7', 1), ('trafficlight_green_l_76', 1), ('trafficlight_green_l_77', 1), ('trafficlight_green_l_9', 1), ('trafficlight_green_m', 31), ('trafficlight_green_ml', 4), ('trafficlight_green_mu', 4), ('trafficlight_green_n', 8917), ('trafficlight_green_n_04', 1), ('trafficlight_green_n_10', 1), ('trafficlight_green_n_12', 1), ('trafficlight_green_n_13', 3), ('trafficlight_green_n_14', 1), ('trafficlight_green_n_2', 2), ('trafficlight_green_n_20', 1), ('trafficlight_green_n_24', 1), ('trafficlight_green_n_25', 2), ('trafficlight_green_n_3', 1), ('trafficlight_green_n_4', 8), ('trafficlight_green_n_5', 16), ('trafficlight_green_n_6', 16), ('trafficlight_green_n_7', 15), ('trafficlight_green_n_8', 16), ('trafficlight_green_n_9', 8), ('trafficlight_green_p', 1680), ('trafficlight_green_p_06', 1), ('trafficlight_green_p_10', 1), ('trafficlight_green_p_12', 1), ('trafficlight_green_p_14', 1), ('trafficlight_green_p_25', 1), ('trafficlight_green_p_28', 2), ('trafficlight_green_p_29', 3), ('trafficlight_green_p_30', 1), ('trafficlight_green_p_31', 3), ('trafficlight_green_p_32', 2), ('trafficlight_green_p_36', 3), ('trafficlight_green_p_5', 1), ('trafficlight_green_p_52', 1), ('trafficlight_green_p_54', 1), ('trafficlight_green_p_55', 3), ('trafficlight_green_p_6', 1), ('trafficlight_green_p_7', 3), ('trafficlight_green_pn', 11), ('trafficlight_green_px', 24), ('trafficlight_green_r', 469), ('trafficlight_green_r_12', 1), ('trafficlight_green_r_13', 4), ('trafficlight_green_r_15', 2), ('trafficlight_green_r_17', 1), ('trafficlight_green_r_22', 1), ('trafficlight_green_r_26', 2), ('trafficlight_green_r_27', 1), ('trafficlight_green_r_28', 1), ('trafficlight_green_r_30', 2), ('trafficlight_green_r_31', 2), ('trafficlight_green_r_32', 4), ('trafficlight_green_r_33', 1), ('trafficlight_green_r_35', 1), ('trafficlight_green_r_4', 1), ('trafficlight_green_r_43', 1), ('trafficlight_green_r_45', 1), ('trafficlight_green_r_5', 1), ('trafficlight_green_r_55', 1), ('trafficlight_green_r_56', 2), ('trafficlight_green_r_58', 1), ('trafficlight_green_r_59', 2), ('trafficlight_green_u', 2031), ('trafficlight_green_u_11', 1), ('trafficlight_green_u_13', 1), ('trafficlight_green_u_15', 2), ('trafficlight_green_u_17', 2), ('trafficlight_green_u_20', 1), ('trafficlight_green_u_22', 2), ('trafficlight_green_u_23', 4), ('trafficlight_green_u_24', 4), ('trafficlight_green_u_25', 2), ('trafficlight_green_u_26', 5), ('trafficlight_green_u_27', 1), ('trafficlight_green_u_28', 2), ('trafficlight_green_u_29', 2), ('trafficlight_green_u_30', 1), ('trafficlight_green_u_32', 2), ('trafficlight_green_u_33', 7), ('trafficlight_green_u_34', 1), ('trafficlight_green_u_35', 1), ('trafficlight_green_u_43', 1), ('trafficlight_green_u_45', 4), ('trafficlight_green_u_47', 3), ('trafficlight_green_u_5', 2), ('trafficlight_green_u_52', 3), ('trafficlight_green_u_53', 2), ('trafficlight_green_u_55', 1), ('trafficlight_green_u_56', 2), ('trafficlight_green_u_58', 5), ('trafficlight_green_u_59', 2), ('trafficlight_green_u_67', 2), ('trafficlight_green_u_68', 1), ('trafficlight_green_u_8', 1), ('trafficlight_green_x', 3071), ('trafficlight_green_x_1', 2), ('trafficlight_green_x_11', 2), ('trafficlight_green_x_14', 2), ('trafficlight_green_x_15', 2), ('trafficlight_green_x_2', 1), ('trafficlight_green_x_20', 1), ('trafficlight_green_x_3', 1), ('trafficlight_green_x_5', 2), ('trafficlight_green_x_55', 2), ('trafficlight_green_x_61', 2), ('trafficlight_green_x_7', 1), ('trafficlight_green_x_8', 1), ('trafficlight_green_x_9', 2), ('trafficlight_grenn_n', 3), ('trafficlight_grenn_u_47', 1), ('trafficlight_off', 3223), ('trafficlight_off_n', 31), ('trafficlight_red', 1), ('trafficlight_red_12', 1), ('trafficlight_red_25_b', 2), ('trafficlight_red_26', 2), ('trafficlight_red_26_b', 1), ('trafficlight_red_27_b', 1), ('trafficlight_red_28_b', 1), ('trafficlight_red_3', 2), ('trafficlight_red_37_b', 4), ('trafficlight_red_38_b', 2), ('trafficlight_red_4', 4), ('trafficlight_red_5', 5), ('trafficlight_red_g', 77), ('trafficlight_red_g_14', 2), ('trafficlight_red_g_16', 1), ('trafficlight_red_g_17', 2), ('trafficlight_red_l', 3622), ('trafficlight_red_l_07', 1), ('trafficlight_red_l_10', 1), ('trafficlight_red_l_11', 1), ('trafficlight_red_l_12', 1), ('trafficlight_red_l_13', 3), ('trafficlight_red_l_15', 3), ('trafficlight_red_l_16', 1), ('trafficlight_red_l_17', 2), ('trafficlight_red_l_18', 3), ('trafficlight_red_l_19', 4), ('trafficlight_red_l_20', 3), ('trafficlight_red_l_21', 6), ('trafficlight_red_l_22', 2), ('trafficlight_red_l_23', 4), ('trafficlight_red_l_26', 2), ('trafficlight_red_l_28', 1), ('trafficlight_red_l_30', 2), ('trafficlight_red_l_31', 3), ('trafficlight_red_l_32', 2), ('trafficlight_red_l_33', 1), ('trafficlight_red_l_4', 1), ('trafficlight_red_l_44', 1), ('trafficlight_red_l_5', 1), ('trafficlight_red_l_6', 12), ('trafficlight_red_l_63', 2), ('trafficlight_red_l_65', 2), ('trafficlight_red_l_66', 1), ('trafficlight_red_l_67', 2), ('trafficlight_red_l_68', 1), ('trafficlight_red_l_7', 6), ('trafficlight_red_l_8', 2), ('trafficlight_red_l_84', 1), ('trafficlight_red_l_9', 4), ('trafficlight_red_m', 23), ('trafficlight_red_n', 17738), ('trafficlight_red_n\t 35', 1), ('trafficlight_red_n.', 1), ('trafficlight_red_n_01', 1), ('trafficlight_red_n_10', 1), ('trafficlight_red_n_11', 3), ('trafficlight_red_n_12', 3), ('trafficlight_red_n_13', 1), ('trafficlight_red_n_17', 3), ('trafficlight_red_n_18', 2), ('trafficlight_red_n_19', 9), ('trafficlight_red_n_2', 1), ('trafficlight_red_n_20', 5), ('trafficlight_red_n_21', 5), ('trafficlight_red_n_22', 2), ('trafficlight_red_n_3', 1), ('trafficlight_red_n_35', 1), ('trafficlight_red_n_38', 1), ('trafficlight_red_n_4', 1), ('trafficlight_red_n_45', 1), ('trafficlight_red_n_48', 1), ('trafficlight_red_n_7', 1), ('trafficlight_red_n_76', 1), ('trafficlight_red_n_8', 2), ('trafficlight_red_p', 2852), ('trafficlight_red_p\t06', 1), ('trafficlight_red_p 40', 1), ('trafficlight_red_p_12', 2), ('trafficlight_red_p_13', 1), ('trafficlight_red_p_14', 5), ('trafficlight_red_p_15', 1), ('trafficlight_red_p_20', 3), ('trafficlight_red_p_22', 1), ('trafficlight_red_p_23', 4), ('trafficlight_red_p_24', 4), ('trafficlight_red_p_25', 1), ('trafficlight_red_p_33', 4), ('trafficlight_red_p_37', 2), ('trafficlight_red_p_39', 1), ('trafficlight_red_p_41', 2), ('trafficlight_red_p_43', 1), ('trafficlight_red_p_44', 1), ('trafficlight_red_p_53', 1), ('trafficlight_red_p_57', 2), ('trafficlight_red_p_6', 5), ('trafficlight_red_p_7', 1), ('trafficlight_red_pn', 14), ('trafficlight_red_px', 45), ('trafficlight_red_px_10', 1), ('trafficlight_red_r', 560), ('trafficlight_red_r_05', 1), ('trafficlight_red_r_13', 5), ('trafficlight_red_r_132', 2), ('trafficlight_red_r_136', 2), ('trafficlight_red_r_137', 5), ('trafficlight_red_r_138', 2), ('trafficlight_red_r_15', 1), ('trafficlight_red_r_17', 1), ('trafficlight_red_r_18', 3), ('trafficlight_red_r_19', 2), ('trafficlight_red_r_24', 1), ('trafficlight_red_r_25', 2), ('trafficlight_red_r_3', 1), ('trafficlight_red_r_4', 7), ('trafficlight_red_r_5', 1), ('trafficlight_red_r_52', 3), ('trafficlight_red_r_53', 4), ('trafficlight_red_r_54', 3), ('trafficlight_red_r_57', 4), ('trafficlight_red_r_58', 3), ('trafficlight_red_u', 3061), ('trafficlight_red_u_01', 1), ('trafficlight_red_u_02', 1), ('trafficlight_red_u_03', 1), ('trafficlight_red_u_05', 2), ('trafficlight_red_u_102', 2), ('trafficlight_red_u_103', 2), ('trafficlight_red_u_104', 5), ('trafficlight_red_u_105', 2), ('trafficlight_red_u_12', 1), ('trafficlight_red_u_13', 4), ('trafficlight_red_u_15', 2), ('trafficlight_red_u_16', 1), ('trafficlight_red_u_17', 2), ('trafficlight_red_u_18', 1), ('trafficlight_red_u_19', 4), ('trafficlight_red_u_21', 4), ('trafficlight_red_u_22', 2), ('trafficlight_red_u_23', 6), ('trafficlight_red_u_24', 7), ('trafficlight_red_u_26', 1), ('trafficlight_red_u_27', 1), ('trafficlight_red_u_30', 1), ('trafficlight_red_u_31', 2), ('trafficlight_red_u_32', 2), ('trafficlight_red_u_33', 1), ('trafficlight_red_u_5', 1), ('trafficlight_red_u_6', 1), ('trafficlight_red_u_63', 1), ('trafficlight_red_u_7', 1), ('trafficlight_red_u_76', 2), ('trafficlight_red_u_77', 1), ('trafficlight_red_u_8', 5), ('trafficlight_red_u_9', 2), ('trafficlight_red_x', 6758), ('trafficlight_red_x_12', 1), ('trafficlight_red_x_14', 3), ('trafficlight_red_x_15', 3), ('trafficlight_red_x_20', 1), ('trafficlight_red_x_22', 3), ('trafficlight_red_x_23', 4), ('trafficlight_red_x_24', 1), ('trafficlight_red_x_29', 4), ('trafficlight_red_x_30', 1), ('trafficlight_red_x_32', 1), ('trafficlight_red_x_33', 1), ('trafficlight_red_x_38', 2), ('trafficlight_red_x_4', 1), ('trafficlight_red_x_52', 1), ('trafficlight_red_x_53', 2), ('trafficlight_red_x_57', 2), ('trafficlight_red_x_72', 2), ('trafficlight_red_x_77', 1), ('trafficlight_red_x_81', 1), ('trafficlight_rede_r_53', 1), ('trafficlight_ren_n', 1), ('trafficlight_ren_p', 1), ('trafficlight_special', 292), ('trafficlight_special_red_n_2', 1), ('trafficlight_yellow_l', 132), ('trafficlight_yellow_n', 862), ('trafficlight_yellow_p', 1), ('trafficlight_yellow_r', 51), ('trafficlight_yellow_u', 87), ('trafficlight_yellow_x', 167), ('trafficlight_yellow_x_5', 1), ('trafficright_green_l_5', 1), ('trafficright_green_p_15', 2), ('trafficright_green_p_20', 1), ('trafficright_green_p_25', 3), ('trafficright_green_p_46', 1), ('trafficright_green_p_52', 1), ('trafficright_green_p_53', 1), ('trafficright_green_p_9', 1), ('trafficright_green_u_15', 1), ('trafficright_green_u_20', 1), ('trafficright_green_u_21', 1), ('trafficright_green_u_22', 1), ('trafficright_green_u_7', 2), ('trafficright_red_l_23', 1), ('trafficright_red_p_12', 3), ('trafficright_red_p_14', 2), ('trafficright_red_p_16', 1), ('trafficright_red_p_2', 1), ('trafficright_red_p_23', 1), ('trafficright_red_p_26', 2), ('trafficright_red_p_33', 1), ('trafficright_red_p_6', 1), ('trafficright_red_p_8', 1), ('trafficright_red_r_11', 1), ('trafficright_red_u_23', 1), ('truck', 46246), ('turn_continue', 3), ('turn_l', 85), ('turn_r', 31), ('unspeed_10', 1), ('unspeed_20', 4), ('unspeed_30', 361), ('unspeed_40', 39), ('unspeed_5', 4), ('unspeed_50', 7), ('unspeed_60', 25), ('unspeed_70', 2), ('uphill_long', 2), ('upspee_30', 1), ('upspeed_10', 372), ('upspeed_100', 17), ('upspeed_120', 7), ('upspeed_15', 239), ('upspeed_20', 1677), ('upspeed_30', 3748), ('upspeed_40', 3557), ('upspeed_5', 212), ('upspeed_50', 864), ('upspeed_55', 1), ('upspeed_60', 2787), ('upspeed_70', 38), ('upspeed_80', 631), ('upspreed_60', 1), ('weight_10', 1), ('weight_13', 1), ('weight_15', 25), ('weight_20', 33), ('weight_25', 2), ('weight_30', 48), ('weight_40', 398), ('weight_49', 4), ('weight_5', 1), ('weight_50', 34), ('weight_55', 156), ('weightr_10', 4), ('weightr_13', 18), ('weightr_14', 24), ('weightr_2.5', 3), ('weightr_40', 14), ('weightr_50', 1), ('weightr_55', 1), ('zebra_crossing', 36924)]

    args.jpg_dir =  args.input_dir + "JPEGImages/"
    args.xml_dir =  args.input_dir + "XML/"

    statistic_classname(args)
