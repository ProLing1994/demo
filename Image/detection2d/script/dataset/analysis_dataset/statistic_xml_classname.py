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

        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库, ET.parse（）打开 xml 文件, parse--"解析"
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

                tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库, ET.parse（）打开 xml 文件, parse--"解析"
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
       
    # 数据集: RM_ADAS_AllInOne
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate_2023_zph/ADAS_AllInOne_MS3_patch/"
    # 数据集: allinone, [('SEWER', 4), ('\\', 1), ('arrow', 1), ('arrow_L', 3), ('arrow_g', 773), ('arrow_gl', 6), ('arrow_gu', 2), ('arrow_l', 4272), ('arrow_lg', 178), ('arrow_lr', 275), ('arrow_lu', 1870), ('arrow_lur', 475), ('arrow_r', 6293), ('arrow_rg', 59), ('arrow_rl', 15), ('arrow_rlu', 11), ('arrow_ru', 6), ('arrow_u', 31109), ('arrow_ug', 146), ('arrow_ul', 1637), ('arrow_ulr', 20), ('arrow_ur', 6313), ('arrow_url', 81), ('bicycle', 14848), ('bicyclist', 19982), ('bus', 48437), ('car', 355810), ('car_big', 180), ('car_big_front', 14741), ('car_big_reg', 65668), ('car_front', 43826), ('car_reg', 261084), ('cat', 3), ('detour_r', 2), ('dog', 172), ('downhill_long', 8), ('downspeed_30', 3), ('downspeed_40', 12), ('downspeed_5', 1), ('downspeed_50', 11), ('downspeed_60', 274), ('downspeed_70', 205), ('height_1.9', 9), ('height_2', 64), ('height_2.2', 7), ('height_2.5', 6), ('height_2.6', 1), ('height_3', 8), ('height_3.2', 2), ('height_3.5', 12), ('height_3.8', 88), ('height_4', 25), ('height_4.1', 7), ('height_4.2', 79), ('height_4.5', 515), ('height_4.7', 38), ('height_5', 949), ('height_5.5', 138), ('height_6', 3), ('motorcycle', 3366), ('motorcyclist', 35728), ('person', 110442), ('person_o', 18976), ('preson', 8), ('sewer', 61184), ('sewer.', 2), ('sign_accident', 112), ('sign_nostop', 36), ('sign_nostopd', 50), ('sign_nostops', 6997), ('sign_stop', 1802), ('sign_stops', 25), ('spupeed_60', 3), ('traffic_red_n', 2), ('trafficlight_greeb_r', 5), ('trafficlight_greeb_x', 3), ('trafficlight_greed_n', 3), ('trafficlight_greed_x', 2), ('trafficlight_green_25', 2), ('trafficlight_green_4', 2), ('trafficlight_green_g', 14), ('trafficlight_green_l', 873), ('trafficlight_green_l_04', 2), ('trafficlight_green_l_1', 2), ('trafficlight_green_l_10', 2), ('trafficlight_green_l_11', 2), ('trafficlight_green_l_13', 2), ('trafficlight_green_l_17', 1), ('trafficlight_green_l_23', 1), ('trafficlight_green_l_3', 1), ('trafficlight_green_l_32', 1), ('trafficlight_green_l_5', 1), ('trafficlight_green_l_55', 1), ('trafficlight_green_l_56', 2), ('trafficlight_green_l_58', 1), ('trafficlight_green_l_59', 1), ('trafficlight_green_l_7', 1), ('trafficlight_green_l_76', 1), ('trafficlight_green_l_77', 1), ('trafficlight_green_l_9', 1), ('trafficlight_green_m', 31), ('trafficlight_green_ml', 4), ('trafficlight_green_mu', 4), ('trafficlight_green_n', 8917), ('trafficlight_green_n_04', 1), ('trafficlight_green_n_10', 1), ('trafficlight_green_n_12', 1), ('trafficlight_green_n_13', 3), ('trafficlight_green_n_14', 1), ('trafficlight_green_n_2', 2), ('trafficlight_green_n_20', 1), ('trafficlight_green_n_24', 1), ('trafficlight_green_n_25', 2), ('trafficlight_green_n_3', 1), ('trafficlight_green_n_4', 8), ('trafficlight_green_n_5', 16), ('trafficlight_green_n_6', 16), ('trafficlight_green_n_7', 15), ('trafficlight_green_n_8', 16), ('trafficlight_green_n_9', 8), ('trafficlight_green_p', 1680), ('trafficlight_green_p_06', 1), ('trafficlight_green_p_10', 1), ('trafficlight_green_p_12', 1), ('trafficlight_green_p_14', 1), ('trafficlight_green_p_25', 1), ('trafficlight_green_p_28', 2), ('trafficlight_green_p_29', 3), ('trafficlight_green_p_30', 1), ('trafficlight_green_p_31', 3), ('trafficlight_green_p_32', 2), ('trafficlight_green_p_36', 3), ('trafficlight_green_p_5', 1), ('trafficlight_green_p_52', 1), ('trafficlight_green_p_54', 1), ('trafficlight_green_p_55', 3), 
    # ('trafficlight_green_p_6', 1), ('trafficlight_green_p_7', 3), ('trafficlight_green_pn', 11), ('trafficlight_green_px', 24), ('trafficlight_green_r', 469), ('trafficlight_green_r_12', 1), ('trafficlight_green_r_13', 4), ('trafficlight_green_r_15', 2), ('trafficlight_green_r_17', 1), ('trafficlight_green_r_22', 1), ('trafficlight_green_r_26', 2), ('trafficlight_green_r_27', 1), ('trafficlight_green_r_28', 1), ('trafficlight_green_r_30', 2), ('trafficlight_green_r_31', 2), ('trafficlight_green_r_32', 4), ('trafficlight_green_r_33', 1), ('trafficlight_green_r_35', 1), ('trafficlight_green_r_4', 1), ('trafficlight_green_r_43', 1), ('trafficlight_green_r_45', 1), ('trafficlight_green_r_5', 1), ('trafficlight_green_r_55', 1), ('trafficlight_green_r_56', 2), ('trafficlight_green_r_58', 1), ('trafficlight_green_r_59', 2), ('trafficlight_green_u', 2031), ('trafficlight_green_u_11', 1), ('trafficlight_green_u_13', 1), ('trafficlight_green_u_15', 2), ('trafficlight_green_u_17', 2), ('trafficlight_green_u_20', 1), ('trafficlight_green_u_22', 2), ('trafficlight_green_u_23', 4), ('trafficlight_green_u_24', 4), ('trafficlight_green_u_25', 2), ('trafficlight_green_u_26', 5), ('trafficlight_green_u_27', 1), ('trafficlight_green_u_28', 2), ('trafficlight_green_u_29', 2), ('trafficlight_green_u_30', 1), ('trafficlight_green_u_32', 2), ('trafficlight_green_u_33', 7), ('trafficlight_green_u_34', 1), ('trafficlight_green_u_35', 1), ('trafficlight_green_u_43', 1), ('trafficlight_green_u_45', 4), ('trafficlight_green_u_47', 3), ('trafficlight_green_u_5', 2), ('trafficlight_green_u_52', 3), ('trafficlight_green_u_53', 2), ('trafficlight_green_u_55', 1), ('trafficlight_green_u_56', 2), ('trafficlight_green_u_58', 5), ('trafficlight_green_u_59', 2), ('trafficlight_green_u_67', 2), ('trafficlight_green_u_68', 1), ('trafficlight_green_u_8', 1), ('trafficlight_green_x', 3071), ('trafficlight_green_x_1', 2), ('trafficlight_green_x_11', 2), ('trafficlight_green_x_14', 2), ('trafficlight_green_x_15', 2), ('trafficlight_green_x_2', 1), ('trafficlight_green_x_20', 1), ('trafficlight_green_x_3', 1), ('trafficlight_green_x_5', 2), ('trafficlight_green_x_55', 2), ('trafficlight_green_x_61', 2), ('trafficlight_green_x_7', 1), ('trafficlight_green_x_8', 1), ('trafficlight_green_x_9', 2), ('trafficlight_grenn_n', 3), ('trafficlight_grenn_u_47', 1), ('trafficlight_off', 3223), ('trafficlight_off_n', 31), ('trafficlight_red', 1), ('trafficlight_red_12', 1), ('trafficlight_red_25_b', 2), ('trafficlight_red_26', 2), ('trafficlight_red_26_b', 1), ('trafficlight_red_27_b', 1), ('trafficlight_red_28_b', 1), ('trafficlight_red_3', 2), ('trafficlight_red_37_b', 4), ('trafficlight_red_38_b', 2), ('trafficlight_red_4', 4), ('trafficlight_red_5', 5), ('trafficlight_red_g', 77), ('trafficlight_red_g_14', 2), ('trafficlight_red_g_16', 1), ('trafficlight_red_g_17', 2), ('trafficlight_red_l', 3622), ('trafficlight_red_l_07', 1), ('trafficlight_red_l_10', 1), ('trafficlight_red_l_11', 1), ('trafficlight_red_l_12', 1), ('trafficlight_red_l_13', 3), ('trafficlight_red_l_15', 3), ('trafficlight_red_l_16', 1), ('trafficlight_red_l_17', 2), ('trafficlight_red_l_18', 3), ('trafficlight_red_l_19', 4), ('trafficlight_red_l_20', 3), ('trafficlight_red_l_21', 6), ('trafficlight_red_l_22', 2), ('trafficlight_red_l_23', 4), ('trafficlight_red_l_26', 2), ('trafficlight_red_l_28', 1), 
    # ('trafficlight_red_l_30', 2), ('trafficlight_red_l_31', 3), ('trafficlight_red_l_32', 2), ('trafficlight_red_l_33', 1), ('trafficlight_red_l_4', 1), ('trafficlight_red_l_44', 1), ('trafficlight_red_l_5', 1), ('trafficlight_red_l_6', 12), ('trafficlight_red_l_63', 2), ('trafficlight_red_l_65', 2), ('trafficlight_red_l_66', 1), ('trafficlight_red_l_67', 2), ('trafficlight_red_l_68', 1), ('trafficlight_red_l_7', 6), ('trafficlight_red_l_8', 2), ('trafficlight_red_l_84', 1), ('trafficlight_red_l_9', 4), ('trafficlight_red_m', 23), ('trafficlight_red_n', 17738), ('trafficlight_red_n\t 35', 1), ('trafficlight_red_n.', 1), ('trafficlight_red_n_01', 1), ('trafficlight_red_n_10', 1), ('trafficlight_red_n_11', 3), ('trafficlight_red_n_12', 3), ('trafficlight_red_n_13', 1), ('trafficlight_red_n_17', 3), ('trafficlight_red_n_18', 2), ('trafficlight_red_n_19', 9), ('trafficlight_red_n_2', 1), ('trafficlight_red_n_20', 5), ('trafficlight_red_n_21', 5), ('trafficlight_red_n_22', 2), ('trafficlight_red_n_3', 1), ('trafficlight_red_n_35', 1), ('trafficlight_red_n_38', 1), ('trafficlight_red_n_4', 1), ('trafficlight_red_n_45', 1), ('trafficlight_red_n_48', 1), ('trafficlight_red_n_7', 1), ('trafficlight_red_n_76', 1), ('trafficlight_red_n_8', 2), ('trafficlight_red_p', 2852), ('trafficlight_red_p\t06', 1), ('trafficlight_red_p 40', 1), ('trafficlight_red_p_12', 2), ('trafficlight_red_p_13', 1), ('trafficlight_red_p_14', 5), ('trafficlight_red_p_15', 1), ('trafficlight_red_p_20', 3), ('trafficlight_red_p_22', 1), ('trafficlight_red_p_23', 4), ('trafficlight_red_p_24', 4), ('trafficlight_red_p_25', 1), ('trafficlight_red_p_33', 4), ('trafficlight_red_p_37', 2), ('trafficlight_red_p_39', 1), ('trafficlight_red_p_41', 2), ('trafficlight_red_p_43', 1), ('trafficlight_red_p_44', 1), ('trafficlight_red_p_53', 1), ('trafficlight_red_p_57', 2), ('trafficlight_red_p_6', 5), ('trafficlight_red_p_7', 1), ('trafficlight_red_pn', 14), ('trafficlight_red_px', 45), ('trafficlight_red_px_10', 1), ('trafficlight_red_r', 560), ('trafficlight_red_r_05', 1), ('trafficlight_red_r_13', 5), ('trafficlight_red_r_132', 2), ('trafficlight_red_r_136', 2), ('trafficlight_red_r_137', 5), ('trafficlight_red_r_138', 2), ('trafficlight_red_r_15', 1), ('trafficlight_red_r_17', 1), ('trafficlight_red_r_18', 3), ('trafficlight_red_r_19', 2), ('trafficlight_red_r_24', 1), ('trafficlight_red_r_25', 2), ('trafficlight_red_r_3', 1), ('trafficlight_red_r_4', 7), ('trafficlight_red_r_5', 1), ('trafficlight_red_r_52', 3), ('trafficlight_red_r_53', 4), ('trafficlight_red_r_54', 3), ('trafficlight_red_r_57', 4), ('trafficlight_red_r_58', 3), ('trafficlight_red_u', 3061), ('trafficlight_red_u_01', 1), ('trafficlight_red_u_02', 1), ('trafficlight_red_u_03', 1), ('trafficlight_red_u_05', 2), ('trafficlight_red_u_102', 2), ('trafficlight_red_u_103', 2), ('trafficlight_red_u_104', 5), ('trafficlight_red_u_105', 2), ('trafficlight_red_u_12', 1), ('trafficlight_red_u_13', 4), ('trafficlight_red_u_15', 2), ('trafficlight_red_u_16', 1), ('trafficlight_red_u_17', 2), ('trafficlight_red_u_18', 1), ('trafficlight_red_u_19', 4), ('trafficlight_red_u_21', 4), ('trafficlight_red_u_22', 2), ('trafficlight_red_u_23', 6), ('trafficlight_red_u_24', 7), ('trafficlight_red_u_26', 1), ('trafficlight_red_u_27', 1), ('trafficlight_red_u_30', 1), ('trafficlight_red_u_31', 2), ('trafficlight_red_u_32', 2),
    # ('trafficlight_red_u_33', 1), ('trafficlight_red_u_5', 1), ('trafficlight_red_u_6', 1), ('trafficlight_red_u_63', 1), ('trafficlight_red_u_7', 1), ('trafficlight_red_u_76', 2), ('trafficlight_red_u_77', 1), ('trafficlight_red_u_8', 5), ('trafficlight_red_u_9', 2), ('trafficlight_red_x', 6758), ('trafficlight_red_x_12', 1), ('trafficlight_red_x_14', 3), ('trafficlight_red_x_15', 3), ('trafficlight_red_x_20', 1), ('trafficlight_red_x_22', 3), ('trafficlight_red_x_23', 4), ('trafficlight_red_x_24', 1), ('trafficlight_red_x_29', 4), ('trafficlight_red_x_30', 1), ('trafficlight_red_x_32', 1), ('trafficlight_red_x_33', 1), ('trafficlight_red_x_38', 2), ('trafficlight_red_x_4', 1), ('trafficlight_red_x_52', 1), ('trafficlight_red_x_53', 2), ('trafficlight_red_x_57', 2), ('trafficlight_red_x_72', 2), ('trafficlight_red_x_77', 1), ('trafficlight_red_x_81', 1), ('trafficlight_rede_r_53', 1), ('trafficlight_ren_n', 1), ('trafficlight_ren_p', 1), ('trafficlight_special', 292), ('trafficlight_special_red_n_2', 1), ('trafficlight_yellow_l', 132), ('trafficlight_yellow_n', 862), ('trafficlight_yellow_p', 1), ('trafficlight_yellow_r', 51), ('trafficlight_yellow_u', 87), ('trafficlight_yellow_x', 167), ('trafficlight_yellow_x_5', 1), ('trafficright_green_l_5', 1), ('trafficright_green_p_15', 2), ('trafficright_green_p_20', 1), ('trafficright_green_p_25', 3), ('trafficright_green_p_46', 1), ('trafficright_green_p_52', 1), ('trafficright_green_p_53', 1), ('trafficright_green_p_9', 1), ('trafficright_green_u_15', 1), ('trafficright_green_u_20', 1), ('trafficright_green_u_21', 1), ('trafficright_green_u_22', 1), ('trafficright_green_u_7', 2), ('trafficright_red_l_23', 1), ('trafficright_red_p_12', 3), ('trafficright_red_p_14', 2), ('trafficright_red_p_16', 1), ('trafficright_red_p_2', 1), ('trafficright_red_p_23', 1), ('trafficright_red_p_26', 2), ('trafficright_red_p_33', 1), ('trafficright_red_p_6', 1), ('trafficright_red_p_8', 1), ('trafficright_red_r_11', 1), ('trafficright_red_u_23', 1), ('truck', 46246), ('turn_continue', 3), ('turn_l', 85), ('turn_r', 31), ('unspeed_10', 1), ('unspeed_20', 4), ('unspeed_30', 361), ('unspeed_40', 39), ('unspeed_5', 4), ('unspeed_50', 7), ('unspeed_60', 25), ('unspeed_70', 2), ('uphill_long', 2), ('upspee_30', 1), ('upspeed_10', 372), ('upspeed_100', 17), ('upspeed_120', 7), ('upspeed_15', 239), ('upspeed_20', 1677), ('upspeed_30', 3748), ('upspeed_40', 3557), ('upspeed_5', 212), ('upspeed_50', 864), ('upspeed_55', 1), ('upspeed_60', 2787), ('upspeed_70', 38), ('upspeed_80', 631), ('upspreed_60', 1), ('weight_10', 1), ('weight_13', 1), ('weight_15', 25), ('weight_20', 33), ('weight_25', 2), ('weight_30', 48), ('weight_40', 398), ('weight_49', 4), ('weight_5', 1), ('weight_50', 34), ('weight_55', 156), ('weightr_10', 4), ('weightr_13', 18), ('weightr_14', 24), ('weightr_2.5', 3), ('weightr_40', 14), ('weightr_50', 1), ('weightr_55', 1), ('zebra_crossing', 36924)]
    # 数据集: allinone_new, [('arrow_g', 3), ('arrow_l', 66), ('arrow_lg', 4), ('arrow_lu', 50), ('arrow_lur', 6), ('arrow_r', 67), ('arrow_u', 292), ('arrow_ug', 4), ('arrow_ur', 89), ('bicycle', 425), ('bicyclist', 571), ('bus', 1849), ('car', 26915), ('car_big_front', 538), ('car_big_reg', 3623), ('car_front', 5428), ('car_reg', 15608), ('dog', 3), ('height_4.5', 5), ('height_5', 14), ('height_5.5', 1), ('motorcycle', 329), ('motorcyclist', 1763), ('negative', 3080), ('person', 5846), ('person_o', 2589), ('sewer', 631), ('sign_accident', 2), ('sign_nostopd', 2), ('sign_nostops', 118), ('sign_stop', 18), ('trafficlight_green_l', 6), ('trafficlight_green_n', 152), ('trafficlight_green_p', 38), ('trafficlight_green_p_2', 1), ('trafficlight_green_p_36', 2), ('trafficlight_green_p_8', 2), ('trafficlight_green_r', 6), ('trafficlight_green_u', 13), ('trafficlight_green_x', 19), ('trafficlight_off', 26), ('trafficlight_red_l', 79), ('trafficlight_red_n', 179), ('trafficlight_red_n12', 1), ('trafficlight_red_n_32', 1), ('trafficlight_red_p', 50), ('trafficlight_red_p_23', 1), ('trafficlight_red_p_3', 1), ('trafficlight_red_p_33', 1), ('trafficlight_red_p_37', 1), ('trafficlight_red_p_6', 1), ('trafficlight_red_r', 17), ('trafficlight_red_u', 63), ('trafficlight_red_x', 83), ('trafficlight_special', 42), ('trafficlight_yellow_n', 1), ('trafficlight_yellow_r', 4), ('truck', 2909), ('unspeed_40', 1), ('upspeed_10', 2), ('upspeed_15', 8), ('upspeed_20', 10), ('upspeed_30', 47), ('upspeed_40', 65), ('upspeed_5', 14), ('upspeed_50', 32), ('upspeed_60', 26), ('weight_10', 1), ('weight_30', 3), ('weight_40', 11), ('weight_49', 1), ('weightr_13', 7), ('weightr_3', 1), ('zebra_crossing', 684)]
    # 数据集: allinone_w_licenseplate, [('arrow', 1), ('arrow_g', 132), ('arrow_l', 333), ('arrow_lg', 29), ('arrow_lr', 45), ('arrow_lu', 182), ('arrow_lur', 139), ('arrow_r', 753), ('arrow_rl', 2), ('arrow_rlu', 11), ('arrow_u', 3659), ('arrow_ug', 16), ('arrow_ul', 72), ('arrow_ulr', 10), ('arrow_ur', 826), ('arrow_url', 27), ('bicycle', 2164), ('bicyclist', 3504), ('bus', 4488), ('car', 44139), ('car_big', 9), ('car_big_front', 916), ('car_big_reg', 5131), ('car_front', 7106), ('car_reg', 28892), ('dog', 21), ('fuzzy_plate', 8261), ('height_1.9', 9), ('height_2', 37), ('height_3', 4), ('height_3.5', 4), ('height_3.8', 48), ('height_4.5', 88), ('height_5', 103), ('height_5.5', 8), ('mote_fuzzy_plate', 144), ('moto_plate', 3), ('motorcycle', 834), ('motorcyclist', 6073), ('painted_plate', 2183), ('person', 28805), ('person_o', 5463), ('plate', 8685), ('preson', 8), ('sewer', 5903), ('sign_accident', 21), ('sign_nostopd', 3), ('sign_nostops', 1574), ('sign_stop', 716), ('trafficlight_green_l', 80), ('trafficlight_green_n', 1657), ('trafficlight_green_p', 663), ('trafficlight_green_p_28', 2), ('trafficlight_green_p_29', 3), ('trafficlight_green_p_30', 1), ('trafficlight_green_p_31', 3), ('trafficlight_green_p_32', 2), ('trafficlight_green_p_52', 1), ('trafficlight_green_r', 30), ('trafficlight_green_u', 262), ('trafficlight_green_x', 796), ('trafficlight_off', 297), ('trafficlight_red_l', 680), ('trafficlight_red_n', 4414), ('trafficlight_red_p', 810), ('trafficlight_red_r', 58), ('trafficlight_red_u', 553), ('trafficlight_red_x', 1625), ('trafficlight_special', 33), ('trafficlight_yellow_l', 24), ('trafficlight_yellow_n', 118), ('trafficlight_yellow_u', 22), ('trafficlight_yellow_x', 22), ('trafficright_green_p_53', 1), ('trafficright_green_p_9', 1), ('tricycle', 43), ('truck', 2552), ('unspeed_30', 34), ('unspeed_40', 9), ('upspeed_10', 233), ('upspeed_15', 61), ('upspeed_20', 431), ('upspeed_30', 968), ('upspeed_40', 419), ('upspeed_5', 3), ('upspeed_50', 206), ('upspeed_60', 250), ('weight_40', 30), ('zebra_crossing', 9492)]
    # 数据集: allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0, [('None', 31), ('bicycle', 106), ('bicyclist', 87), ('bridge_4.5m', 21), ('bridge_5m', 9), ('bridge_h', 247), ('bridge_m', 42), ('bus', 189), ('car', 2885), ('car_big_front', 45), ('car_big_reg', 426), ('car_front', 627), ('car_reg', 1915), ('downhill_long', 2), ('licence', 90), ('licence_f', 1202), ('licence_o', 87), ('motorcycle', 304), ('motorcyclist', 1097), ('person', 985), ('person_o', 134), ('sign_accident', 3), ('sign_nostops', 112), ('sign_stop', 8), ('trafficlight_green_n', 183), ('trafficlight_green_u', 10), ('trafficlight_green_x', 55), ('trafficlight_off', 4), ('trafficlight_red_l', 17), ('trafficlight_red_n', 121), ('trafficlight_red_r', 4), ('trafficlight_red_x', 23), ('trafficlight_special', 1), ('trafficlight_yellow_n', 1), ('trafficlight_yellow_u', 15), ('tricycle', 7), ('truck', 372), ('upspeed_15', 2), ('upspeed_20', 65), ('upspeed_30', 26), ('upspeed_40', 18), ('upspeed_60', 20), ('upspeed_80', 3), ('weight_40', 9), ('zebra_crossing', 354)]
    # 数据集: allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0_5M_Backlight, [('bicycle', 114), ('bicyclist', 60), ('bridge_4.5m', 1), ('bridge_4m', 1), ('bridge_5.5m', 1), ('bridge_5m', 43), ('bridge_h', 102), ('bridge_m', 66), ('bus', 309), ('car', 4153), ('car_big_front', 105), ('car_big_reg', 418), ('car_front', 668), ('car_reg', 2909), ('downspeed_60', 1), ('licence', 23), ('licence_f', 3025), ('licence_o', 136), ('motorcycle', 246), ('motorcyclist', 1582), ('person', 790), ('person_o', 296), ('sign_nostop', 3), ('sign_nostopd', 1), ('sign_nostops', 110), ('sign_stop', 6), ('trafficlight_green_l', 8), ('trafficlight_green_n', 166), ('trafficlight_green_u', 8), ('trafficlight_green_x', 31), ('trafficlight_red_l', 2), ('trafficlight_red_n', 128), ('trafficlight_red_u', 4), ('trafficlight_red_x', 46), ('trafficlight_special', 3), ('trafficlight_yellow_n', 1), ('trafficlight_yellow_x', 2), ('truck', 428), ('turn_l', 6), ('turn_r', 2), ('unspeed_30', 8), ('upspeed_15', 1), ('upspeed_20', 12), ('upspeed_30', 37), ('upspeed_40', 24), ('upspeed_50', 10), ('upspeed_60', 6), ('upspeed_80', 3), ('weight_30', 1), ('weight_40', 6), ('weight_49', 2), ('weight_55', 6), ('zebra_crossing', 302)]
    # 数据集: allinone_w_licenseplate_2023_zph/ADAS_ADPLUS2.0_NearPerson, [('bicycle', 2), ('bicyclist', 43), ('car', 2653), ('car_big_front', 20), ('car_big_reg', 2), ('car_front', 1087), ('car_reg', 1095), ('licence', 239), ('licence_f', 1424), ('licence_o', 90), ('motorcyclist', 217), ('person', 12916), ('person_o', 18), ('truck', 21)]
    # 数据集: allinone_w_licenseplate_2023_zph/ADAS_AllInOne_Backlight_AbnormalVehicle, [('bicycle', 274), ('bicyclist', 306), ('bridge_2m', 4), ('bridge_3.4m', 1), ('bridge_3.5m', 1), ('bridge_4.5m', 2), ('bridge_4m', 2), ('bridge_5m', 1), ('bridge_h', 231), ('bridge_m', 23), ('bus', 406), ('car', 9299), ('car_big_front', 356), ('car_big_reg', 1127), ('car_front', 1558), ('car_reg', 5878), ('downspeed_60', 2), ('downspeed_80', 1), ('licence', 26), ('licence_f', 6732), ('licence_o', 69), ('motorcycle', 205), ('motorcyclist', 1490), ('person', 2028), ('person_o', 796), ('sign_nostop', 2), ('sign_nostops', 61), ('sign_stop', 6), ('trafficlight_green_l', 18), ('trafficlight_green_n', 457), ('trafficlight_green_r', 7), ('trafficlight_green_u', 15), ('trafficlight_green_x', 47), ('trafficlight_red_g', 2), ('trafficlight_red_l', 62), ('trafficlight_red_n', 451), ('trafficlight_red_r', 2), ('trafficlight_red_u', 8), ('trafficlight_red_x', 65), ('trafficlight_special', 1), ('trafficlight_yellow_l', 5), ('trafficlight_yellow_n', 26), ('trafficlight_yellow_x', 2), ('tricycle', 2), ('truck', 1700), ('turn_l', 3), ('turn_r', 2), ('unspeed_100', 3), ('unspeed_50', 2), ('unspeed_60', 5), ('unspeed_80', 1), ('upspeed_10', 1), ('upspeed_15', 2), ('upspeed_20', 5), ('upspeed_30', 44), ('upspeed_35', 1), ('upspeed_40', 36), ('upspeed_5', 13), ('upspeed_50', 44), ('upspeed_60', 40), ('upspeed_70', 2), ('upspeed_80', 4), ('weight_10', 1), ('weight_20', 3), ('weight_30', 3), ('weight_40', 1), ('weight_49', 1), ('weightr_13', 5), ('zebra_crossing', 586)]
    # 数据集: allinone_w_licenseplate_2023_zph/ADAS_AllInOne_MS1, [('bicycle', 654), ('bicyclist', 568), ('bus', 1286), ('car', 19878), ('car_big_front', 277), ('car_big_reg', 2397), ('car_front', 2056), ('car_reg', 14941), ('downhill_long', 1), ('downspeed_110', 1), ('downspeed_40', 2), ('downspeed_50', 6), ('downspeed_60', 9), ('downspeed_80', 5), ('downspeed_90', 1), ('licence', 3001), ('licence_f', 6637), ('licence_o', 470), ('motorcycle', 944), ('motorcyclist', 2691), ('person', 5420), ('person_o', 1235), ('sign_accident', 1), ('sign_nostop', 19), ('sign_nostops', 401), ('sign_stop', 38), ('tricycle', 22), ('truck', 2456), ('turn_l', 2), ('turn_r', 3), ('unspeed_120', 4), ('upspeed_10', 7), ('upspeed_100', 8), ('upspeed_15', 16), ('upspeed_20', 48), ('upspeed_30', 181), ('upspeed_35', 5), ('upspeed_40', 189), ('upspeed_5', 11), ('upspeed_50', 72), ('upspeed_60', 127), ('upspeed_70', 24), ('upspeed_80', 45), ('weight_20', 1), ('weight_30', 7), ('weight_40', 26), ('weight_49', 36), ('weight_55', 5), ('weightr_13', 21), ('weightr_14', 31), ('zebra_crossing', 2160)]
    # 数据集: allinone_w_licenseplate_2023_zph/ADAS_AllInOne_Rainy_Night, [('bicycle', 248), ('bicyclist', 76), ('bridge_4.5m', 9), ('bridge_4m', 12), ('bridge_5m', 124), ('bridge_h', 262), ('bridge_m', 114), ('bus', 295), ('car', 11394), ('car_big_front', 55), ('car_big_reg', 696), ('car_front', 473), ('car_reg', 8772), ('downspeed_40', 9), ('downspeed_50', 18), ('downspeed_60', 6), ('downspeed_90', 1), ('licence', 370), ('licence_f', 8652), ('licence_o', 263), ('motorcycle', 258), ('motorcyclist', 658), ('person', 1815), ('person_o', 443), ('sign_accident', 3), ('sign_nostop', 5), ('sign_nostops', 219), ('sign_stop', 10), ('trafficlight_green_l', 16), ('trafficlight_green_n', 338), ('trafficlight_green_u', 19), ('trafficlight_green_x', 69), ('trafficlight_red_l', 16), ('trafficlight_red_n', 181), ('trafficlight_red_r', 17), ('trafficlight_red_u', 1), ('trafficlight_red_x', 33), ('trafficlight_special', 19), ('trafficlight_yellow_l', 3), ('trafficlight_yellow_n', 43), ('trafficlight_yellow_r', 1), ('tricycle', 2), ('truck', 715), ('turn_l', 1), ('turn_r', 2), ('unspeed_40', 2), ('unspeed_50', 2), ('unspeed_60', 3), ('upspeed_15', 10), ('upspeed_20', 41), ('upspeed_30', 144), ('upspeed_35', 12), ('upspeed_40', 136), ('upspeed_5', 1), ('upspeed_50', 36), ('upspeed_60', 127), ('upspeed_70', 12), ('upspeed_80', 33), ('weight_10', 1), ('weight_30', 6), ('weight_40', 33), ('weight_49', 23), ('weight_55', 7), ('weightr_13', 5), ('weightr_14', 3), ('zebra_crossing', 911)]
    # 数据集: allinone_w_licenseplate_2023_zph/ADAS_Night_Highway_Backlight, ('bridge_5m', 12), ('bridge_h', 28), ('bridge_m', 9), ('bus', 3), ('car', 2173), ('car_big_front', 437), ('car_big_reg', 3087), ('car_front', 196), ('car_reg', 1932), ('downspeed_100', 5), ('downspeed_60', 32), ('downspeed_70', 1), ('downspeed_80', 1), ('downspeed_90', 1), ('licence', 4), ('licence_f', 2537), ('licence_o', 22), ('sign_nostops', 19), ('trafficlight_red_n', 1), ('truck', 3730), ('turn_r', 4), ('unspeed_100', 1), ('unspeed_120', 1), ('unspeed_60', 8), ('upspeed90', 4), ('upspeed_100', 35), ('upspeed_120', 32), ('upspeed_40', 42), ('upspeed_60', 33), ('upspeed_80', 19)]
    # 数据集: allinone_w_licenseplate_2023_zph/ADAS_AbnormalVehicle, [('bicycle', 158), ('bicyclist', 79), ('bridge_4.5m', 5), ('bridge_5.3m', 1), ('bridge_5m', 28), ('bridge_h', 1075), ('bridge_m', 45), ('bus', 787), ('car', 62572), ('car_big', 3), ('car_big_front', 4780), ('car_big_reg', 17443), ('car_front', 19773), ('car_reg', 35843), ('downhill_continue', 4), ('downspeed_40', 1), ('downspeed_70', 1), ('downspeed_china_40', 2), ('downspeed_china_50', 8), ('downspeed_china_60', 5), ('downspeed_china_blur', 2), ('heigh_2.6m', 18), ('heigh_2.8m', 4), ('heigh_3.8m', 24), ('heigh_4.5m', 3), ('heigh_4m', 1), ('heigh_5m', 8), ('heigh_china_2.5m', 3), ('heigh_china_4.5m', 2), ('heigh_china_5.5m', 1), ('heigh_china_5m', 25), ('licence', 3167), ('licence_f', 19175), ('licence_o', 261), ('motorcycle', 75), ('motorcyclist', 753), ('person', 1983), ('person_o', 443), ('rafficlight_4_3_l', 14), ('sign_accident', 3), ('sign_nostop_china', 1), ('sign_nostopd', 2), ('sign_nostops', 150), ('sign_stop', 345), ('sign_stop_china', 10), ('trafficlight_4_1_2_l', 4), ('trafficlight_4_1_I', 5), ('trafficlight_4_1_l', 3), ('trafficlight_4_1_n', 103), ('trafficlight_4_1_n_4_l', 55), ('trafficlight_4_3_l', 13), ('trafficlight_4_3_l_4_n', 23), ('trafficlight_4_3_n', 206), ('trafficlight_4_3_n_4_l', 105), ('trafficlight_4_4_l', 2), ('trafficlight_4_4_n', 33), ('trafficlight_4_off', 32), ('trafficlight_5_1_l_2_n_5_n', 1), ('trafficlight_5_1_n', 13), ('trafficlight_5_1_n_4_l', 1), ('trafficlight_5_2_n', 1), ('trafficlight_5_3_n', 46), ('trafficlight_5_5_n', 107), ('trafficlight_5_off', 17), ('trafficlight_green_l', 143), ('trafficlight_green_n', 3234), ('trafficlight_green_r', 5), ('trafficlight_green_u', 206), ('trafficlight_green_x', 260), ('trafficlight_off', 277), ('trafficlight_red_g', 3), ('trafficlight_red_l', 159), ('trafficlight_red_n', 1139), ('trafficlight_red_r', 3), ('trafficlight_red_u', 6), ('trafficlight_red_x', 65), ('trafficlight_share0_green_n', 3), ('trafficlight_share0_red_n', 6), ('trafficlight_share1_green_l_green_n', 54), ('trafficlight_share1_green_l_red_n', 3), ('trafficlight_share1_red_l_green_n', 35), ('trafficlight_share1_red_n', 26), ('trafficlight_share2_green_l_green_n', 26), ('trafficlight_share2_green_n', 124), ('trafficlight_share2_red_l_green_n', 6), ('trafficlight_special', 37), ('trafficlight_yellow_l', 9), ('trafficlight_yellow_n', 101), ('trafficlight_yellow_u', 1), ('trafficlight_yellow_x', 8), ('tricycle', 1), ('tricycle_reg', 18), ('tricyclist', 18), ('truck', 25056), ('unspeed_50', 16), ('unspeed_china_30', 1), ('upseed_65', 2), ('upspeed90', 27), ('upspeed_10', 3), ('upspeed_100', 72), ('upspeed_110', 9), ('upspeed_120', 6), ('upspeed_15', 111), ('upspeed_20', 10), ('upspeed_25', 1), ('upspeed_30', 130), ('upspeed_35', 7), ('upspeed_40', 15), ('upspeed_45', 24), ('upspeed_50', 71), ('upspeed_55', 42), ('upspeed_60', 42), ('upspeed_65', 1), ('upspeed_70', 40), ('upspeed_80', 29), ('upspeed_china_15', 1), ('upspeed_china_20', 2), ('upspeed_china_30', 7), ('upspeed_china_40', 33), ('upspeed_china_50', 17), ('upspeed_china_60', 19), ('upspeed_china_70', 8), ('upspeed_china_80', 7), ('upspeed_china_blur', 17), ('upspeed_usa_100_blur', 5), ('upspeed_usa_35', 94), ('upspeed_usa_45', 2), ('upspeed_usa_55', 15), ('upspeed_usa_55_blur', 16), ('upspeed_usa_65', 21), ('upspeed_usa_65_blur', 3), ('upspeed_usa_70', 8), ('upspeed_usa_blur', 7), ('upspeed_usa_ntc_55', 7), ('upspeedy_usa_50', 5), ('upspeedy_usa_50_blur', 1), ('upspeedy_usa_70', 6), ('weight_49', 7), ('weightr_13', 1), ('weightr_14', 4), ('zebra_crossing', 1978)]
    # 数据集: allinone_w_licenseplate_2023_zph/ADAS_AllInOne_AbnormalVehicle, ][('bicycle', 130), ('bicyclist', 81), ('bridge_4.3m', 10), ('bridge_4.5m', 8), ('bridge_5m', 21), ('bridge_h', 178), ('bridge_m', 103), ('bus', 752), ('car', 9986), ('car_big_front', 227), ('car_big_reg', 4721), ('car_front', 672), ('car_reg', 8278), ('downspeed_40', 1), ('heigh_2.6m', 5), ('heigh_2.8m', 2), ('heigh_4.5m', 6), ('heigh_5.5m', 17), ('heigh_5m', 17), ('licence', 1348), ('licence_f', 6185), ('licence_o', 220), ('motorcycle', 238), ('motorcyclist', 404), ('person', 743), ('person_o', 194), ('sign_accident', 10), ('sign_nostop', 2), ('sign_nostopd', 5), ('sign_nostops', 109), ('sign_stop', 39), ('trafficlight_green_l', 5), ('trafficlight_green_n', 37), ('trafficlight_green_u', 63), ('trafficlight_green_x', 12), ('trafficlight_off', 5), ('trafficlight_red_l', 5), ('trafficlight_red_n', 139), ('trafficlight_red_r', 4), ('trafficlight_red_x', 16), ('trafficlight_yellow_n', 11), ('tricycle', 11), ('tricycle_reg', 4), ('tricyclist', 5), ('truck', 4626), ('unspeed_30', 3), ('unspeed_40', 20), ('unspeed_60', 1), ('unspeed_80', 1), ('upspeed_100', 6), ('upspeed_15', 1), ('upspeed_20', 30), ('upspeed_30', 77), ('upspeed_40', 23), ('upspeed_5', 2), ('upspeed_50', 5), ('upspeed_60', 87), ('upspeed_70', 7), ('upspeed_80', 21), ('weight_30', 2), ('weight_40', 3), ('weight_49', 67), ('weightr_13', 23), ('weightr_14', 41), ('zebra_crossing', 402)]
    # 数据集: allinone_w_licenseplate_2023_zph/ADAS_AllInOne_MS3_patch, [('bicycle', 392), ('bicyclist', 329), ('bridge_2.4m', 1), ('bridge_2m', 2), ('bridge_3.2m', 1), ('bridge_3.5m', 2), ('bridge_3.8m', 1), ('bridge_4.2m', 4), ('bridge_4.3m', 2), ('bridge_4.5m', 11), ('bridge_4.8m', 4), ('bridge_4m', 4), ('bridge_5.5m', 10), ('bridge_5m', 71), ('bridge_h', 185), ('bridge_m', 107), ('bus', 1378), ('car', 16640), ('car_big_front', 451), ('car_big_reg', 2567), ('car_front', 2389), ('car_reg', 12585), ('downhill_long', 1), ('downspeed_40', 1), ('downspeed_50', 5), ('downspeed_60', 5), ('downspeed_70', 2), ('heigh_2.3m', 2), ('heigh_2.4m', 1), ('heigh_2.6m', 2), ('heigh_2m', 3), ('heigh_3.2m', 1), ('heigh_3.5m', 2), ('heigh_3.8m', 1), ('heigh_4.2m', 3), ('heigh_4.5m', 10), ('heigh_4.8m', 3), ('heigh_4m', 2), ('heigh_5.5m', 15), ('heigh_5m', 69), ('licence', 2321), ('licence_f', 8161), ('licence_o', 445), ('motorcycle', 462), ('motorcyclist', 2164), ('person', 4355), ('person_o', 616), ('sign_accident', 2), ('sign_nostop', 23), ('sign_nostopd', 5), ('sign_nostops', 398), ('sign_stop', 23), ('trafficlight_green_g', 3), ('trafficlight_green_l', 41), ('trafficlight_green_n', 642), ('trafficlight_green_r', 15), ('trafficlight_green_u', 142), ('trafficlight_green_x', 85), ('trafficlight_off', 35), ('trafficlight_red_l', 80), ('trafficlight_red_n', 352), ('trafficlight_red_r', 21), ('trafficlight_red_u', 49), ('trafficlight_red_x', 53), ('trafficlight_special', 8), ('trafficlight_yellow_g', 1), ('trafficlight_yellow_l', 12), ('trafficlight_yellow_n', 64), ('trafficlight_yellow_r', 3), ('trafficlight_yellow_u', 6), ('trafficlight_yellow_x', 3), ('tricycle', 14), ('tricycle_reg', 33), ('tricyclist', 47), ('truck', 1998), ('turn_l', 1), ('unspeed_120', 1), ('unspeed_20', 1), ('unspeed_30', 7), ('unspeed_50', 1), ('unspeed_60', 1), ('upspeed_10', 2), ('upspeed_15', 25), ('upspeed_20', 62), ('upspeed_30', 155), ('upspeed_35', 1), ('upspeed_40', 194), ('upspeed_5', 8), ('upspeed_50', 71), ('upspeed_60', 107), ('upspeed_70', 6), ('upspeed_80', 22), ('weight_20', 1), ('weight_30', 6), ('weight_40', 23), ('weight_49', 35), ('weight_55', 7), ('weightr_13', 22), ('weightr_14', 29), ('zebra_crossing', 1166)]
    # 数据集: allinone_w_licenseplate_2023_zph_new_style/ADAS_AllInOne_New_Test, [('arrow_g', 175), ('arrow_l', 532), ('arrow_lg', 36), ('arrow_lr', 124), ('arrow_lu', 167), ('arrow_lur', 76), ('arrow_r', 1182), ('arrow_rg', 2), ('arrow_u', 3856), ('arrow_ug', 14), ('arrow_ul', 76), ('arrow_ur', 769), ('bicycle', 898), ('bicyclist', 2608), ('bus', 6316), ('car', 57187), ('car_big', 1), ('car_big_front', 816), ('car_big_reg', 8510), ('car_front', 3720), ('car_reg', 44546), ('dog', 15), ('downspeed_10', 1), ('downspeed_20', 42), ('downspeed_30', 19), ('downspeed_40', 10), ('downspeed_50', 30), ('downspeed_60', 18), ('downspeed_70', 18), ('downspeed_80', 1), ('f', 1), ('motorcy', 2), ('motorcycle', 286), ('motorcyclist', 3466), ('person', 18336), ('person_o', 2511), ('sign_nostop', 10), ('sign_nostops', 1082), ('sign_stop', 254), ('tricycle', 22), ('truck', 4407), ('unspeed_10', 21), ('unspeed_15', 5), ('unspeed_20', 19), ('unspeed_30', 92), ('unspeed_40', 46), ('unspeed_50', 13), ('unspeed_60', 1), ('unspeed_80', 3), ('upspeed-40', 111), ('upspeed_10', 337), ('upspeed_15', 120), ('upspeed_2', 2), ('upspeed_20', 421), ('upspeed_30', 868), ('upspeed_40', 420), ('upspeed_5', 26), ('upspeed_50', 363), ('upspeed_60', 91), ('upspeed_70', 20), ('upspeed_80', 70), ('weight_20', 1), ('weight_40', 87), ('weightr_40', 31), ('weightr_55', 1), ('zebra_crossing', 6410)]
    # 数据集: allinone_w_licenseplate_2023_zph_new_style/ADAS_VehicleTailRegression, [('bicycle', 51), ('bicyclist', 7), ('bridge_h', 13), ('bus', 396), ('car', 31110), ('car_big_front', 2451), ('car_big_reg', 7764), ('car_front', 11024), ('car_reg', 15477), ('downspeed_70', 1), ('licence', 232), ('licence_f', 647), ('licence_o', 18), ('motorcycle', 4), ('motorcyclist', 62), ('person', 1214), ('person_o', 206), ('sign_accident', 2), ('sign_stop', 73), ('trafficlight_green_n', 1), ('trafficlight_green_x', 130), ('trafficlight_red_g', 1), ('trafficlight_red_x', 7), ('trafficlight_yellow_u', 1), ('tricycle_reg', 1), ('truck', 12063), ('upseed_65', 2), ('upspeed_100', 12), ('upspeed_120', 5), ('upspeed_30', 4), ('zebra_crossing', 6)]

    # 数据集: ZF_Europe, ADAS 视角
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/hardNeg/"
    # 数据集: england, [('car', 156956), ('face', 4087), ('licenseplate', 72909), ('person', 24889), ('person_o', 17738)]
    # 数据集: england_1080p, [('car', 162837), ('face', 731), ('licence', 3932), ('licence_f', 47383), ('licence_o', 2207), ('person', 15254), ('person_o', 7973)]
    # 数据集: france, [('car', 168436), ('face', 4663), ('face.', 1), ('lecense', 357), ('lecense_f', 1637), ('lecense_o', 30), ('licence', 22969), ('licence_F', 55), ('licence_f', 55508), ('licence_o', 5250), ('person', 44017), ('person_o', 8060)]
    # 数据集: italy, [('car', 22473), ('face', 84), ('licence', 4027), ('licence_f', 6788), ('licence_o', 666), ('person', 2182), ('person_o', 806)]
    # 数据集: netherlands, [('car', 146085), ('face', 1467), ('fance', 1), ('licence', 6280), ('licence_f', 50968), ('licence_o', 4451), ('person', 26884), ('person_o', 3823), ('rail', 3)]
    # 数据集: moni, [('car', 23631), ('face', 22293), ('licence', 769), ('licence_f', 13500), ('licence_o', 747), ('person', 56504), ('person_o', 6134)]
    # 数据集: moni_0415, [('car', 7446), ('face', 16092), ('licence', 1228), ('licence_f', 4369), ('licence_o', 677), ('person', 36809), ('person_o', 615), ('roads', 1)]
    # 数据集: hardNeg, [('car', 89750), ('car_front', 7792), ('car_reg', 18801), ('face', 2067), ('licence', 5145), ('licence_f', 22953), ('licence_o', 992), ('person', 4582), ('person_o', 4119)]   

    # 数据集: LicensePlate_detection
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_detection/China/"
    # 数据集: China：car 机动车, license_plate 清晰车牌, [(' upspeed_20', 2), (' upspeed_5', 1), ('bicycle', 8978), ('bicycle ', 1), ('car', 165480), ('downspeed_40', 3), ('downspeed_60', 46), ('downspeed_70', 37), ('height_2.2', 4), ('height_2.5', 6), ('height_2.6', 1), ('height_3', 1), ('height_3.8', 4), ('height_4', 2), ('height_4.0', 10), ('height_4.2', 4), ('height_4.5', 167), ('height_4.7', 5), ('height_4.8', 3), ('height_5', 140), ('height_5.0', 127), ('height_5.5', 49), ('license_plate', 35630), ('motorcycle', 8589), ('person', 40085), ('person_o', 9257), ('sign_stop', 303), ('traffic_sign', 1), ('trafficlight_green_l', 266), ('trafficlight_green_n', 1321), ('trafficlight_green_r', 110), ('trafficlight_green_u', 466), ('trafficlight_green_u ', 1), ('trafficlight_green_x', 541), ('trafficlight_off', 654), ('trafficlight_red_l', 803), ('trafficlight_red_l ', 1), ('trafficlight_red_n', 1537), ('trafficlight_red_n ', 3), ('trafficlight_red_r', 29), ('trafficlight_red_u', 467), ('trafficlight_red_x', 799), ('trafficlight_special', 42), ('trafficlight_yellow_l', 33), ('trafficlight_yellow_n', 221), ('trafficlight_yellow_r', 7), ('trafficlight_yellow_u', 33), ('trafficlight_yellow_x', 39), ('unspeed_30', 52), ('upspeed_10', 26), ('upspeed_15', 35), ('upspeed_20', 325), ('upspeed_30', 914), ('upspeed_40', 870), ('upspeed_5', 53), ('upspeed_50', 241), ('upspeed_60', 711), ('upspeed_70', 19), ('upspeed_80', 148), ('weight_15', 5), ('weight_20', 13), ('weight_30', 21), ('weight_40', 170), ('weight_49', 3), ('weight_50', 12), ('weight_55', 34), ('weightr_10', 2), ('weightr_13', 6), ('weightr_14', 8), ('zebra_crossing', 9718)
    # 数据集: China_6mm：car 机动车, license_plate 清晰车牌, {'car': 6206, 'license_plate': 2665}
    # 数据集: Europe：license_plate 清晰车牌, {'license_plate': 24836}
    # 数据集: Mexico：car 机动车, license_plate 清晰车牌, {'car': 47367, 'license_plate': 17805}

    # 数据集: RM_C27_detection
    # args.input_dir = "/yuanhuan/data/image/RM_C27_detection/original/zd_c27_20200209_20201125/"
    # args.input_dir = "/yuanhuan/data/image/RM_C27_detection/original/zd_c27_20200217_20200324_car/"
    # args.input_dir = "/yuanhuan/data/image/RM_C27_detection/original/zd_c27_20190713_20200301_plate/"
    # args.input_dir = "/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil/Brazil_all/"
    # args.input_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_new_style/shate_20230308/"
    # args.input_dir = "/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_new_style/shate_20230309/"
    # 数据集: zd_c27_20200209_20201125, license_plate 清晰车牌, [('car', 251234), ('license_plate', 163165)]
    # 数据集: zd_c27_20200217_20200324_car, license_plate 清晰车牌, [('car', 18443), ('license_plate', 620)]
    # 数据集: zd_c27_20190713_20200301_plate, license_plate 清晰车牌, 
    # 数据集: Brazil_all, [('bicyclist', 38), ('bus', 15165), ('car', 433038), ('cover-motorplate', 2905), ('cover-plaet', 37229), ('fuzzy-motorplate', 22150), ('fuzzy-plate', 84662), ('kind', 1), ('lince-motorplate', 19846), ('lince-plate', 113262), ('motorcycle', 27374), ('motorcyclist', 31297), ('num', 3), ('truck', 34930)]
    # 数据集: shate_20230308, [('bicycle', 7), ('bus', 1205), ('car', 89410), ('car_big', 5), ('license', 40107), ('motorcycle', 98), ('tricycle', 1), ('truck', 7082)]

    # 数据集: RM_R151_detection
    # 数据集：仅标注车牌，清晰、模糊车牌均统一标注
    # args.input_dir = "/yuanhuan/data/image/RM_R151_detection/original/FW_230612_p1_383/"
    # 数据集: C53_AB_WA_20230426_FN_detline_merge_test, [('license', 184)]
    # 数据集: C53_ITA_WA_20230324_detonly_merge, [('license', 823)]
    # 数据集: C53_NLD_WA_20230307_detonly_merge, [('license', 4475)]
    # 数据集: C53AB_WA_20221215_detonly_merge_plate, [('license', 13058)]
    # 数据集: C53AB_WA_20221216_detonly_merge, [('license', 11743)]
    # 数据集: C53AB_WA_20221227_detonly_merge, [('license', 3775)]
    # 数据集: C53_20220630_WA, [('license', 2356)]
    # 数据集: C53_20220701_WA, [('license', 465)]
    # 数据集: C53_20220707_WA, [('license', 1481)]
    # 数据集: C53_20220729_WA, [('license', 1149)]
    # 数据集: C53_AB_WA_20230412_FP_detonly_merge, [('license', 1395)]
    # 数据集: C53_AB_WA_20230426_FN_detline_merge, [('license', 1799)]
    # 数据集: C53_AB_WA_20230525_detline_merge, [('license', 3016)]
    # 数据集: C53_CHN_FW_20230426_FN_detline_merge, [('license', 4914)]
    # 数据集: C53_CHN_WA_20230426_FN_detline_merge, [('license', 8369)]
    # 数据集: C53_CHN_WA_20230525_detline_merge, [('license', 6179)]
    # 数据集: C53_GBR_WA_20230324_polyline_merge, [('license', 855)]
    # 数据集: C53_NLD_WA_20230324_daetonly_merge, [('license', 83)]
    # 数据集: C53AB_20221214_WA, [('license', 2224)]
    # 数据集: C53AB_WA_20221214_detonly_merge_extra_p, [('license', 4887)]
    # 数据集: C53AB_WA_20221214_detonly_merge_extra_p0, [('license', 1871)]
    # 数据集: C53AB_WA_20221227_detonly_empty_merge, [('license', 280)]
    # 数据集: C53Germany_WA_20230124_detonly_merge_p, [('license', 4696)]
    # 数据集: C53AB_LF_20221214_detonly_merge_extra_p1_p2_2_389, [('license', 1813)]
    # 数据集: C53AB_LF_20221214_detonly_merge_extra_p1_p2_388, [('license', 565)]
    # 数据集: C53AB_LF_20221215_16_pla_386, [('license', 22048)]
    # 数据集: FW_230612_p1_383, [('license', 8596)]

    # 数据集: RM_BSD
    # args.input_dir = "/yuanhuan/data/image/RM_BSD/wideangle_2022_2023/"
    # 数据集: bsd_20220425_20220512, [('bicycle', 724), ('bus', 3821), ('car', 7991), ('motorcycle', 1604), ('person', 5963), ('truck', 2732)]
    # 数据集: wideangle_2022_2023, [('bicyclist', 9654), ('blank', 1), ('bus', 1356), ('bus_front', 1), ('car', 77930), ('cyclist', 1358), ('empty', 1803), ('green_belts', 8102), ('motorcyclist', 22125), ('person', 133929), ('rail', 12507), ('roadside', 17857), ('scooter_rider', 1), ('truck', 4321)]

    # 数据集: RM_C28_detection
    # args.input_dir = "/yuanhuan/data/image/RM_C28_detection/america_new/"
    # zhongdong：[('car', 510565)]
    # safezone：[('car', 217765)]
    # china：[('CAR', 14), ('car', 789714), ('car ', 1037), ('person', 8107)]
    # canada：[('car', 110763), ('person', 11789)]
    # america：[('car', 403920)]
    # america_new：[('bus', 2758), ('car', 29962), ('truck', 1130)]

    # 数据集: ZG_ZHJYZ_detection, 包含 car\bus\truck\plate\fuzzy_plate, 龙门架视角 
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/original/anhuihuaibeigaosu_night_diguangzhao/"
    # 数据集: jiayouzhan, [('bicyclist', 13), ('bus', 685), ('car', 12295), ('fuzzy_plate', 1792), ('motorcyclist', 44), ('plate', 9800), ('roi_ignore_plate', 227), ('tricycle', 17), ('truck', 180)]
    # 数据集: jiayouzhan_5M, [('bicyclist', 8), ('bus', 489), ('car', 7774), ('fuzzy_plate', 1158), ('motorcyclist', 52), ('painted_plate', 175), ('plate', 4014), ('truck', 22)]
    # 数据集: sandaofangxian, [('bus', 1784), ('car', 9226), ('fuzzy_plate', 4421), ('painted_plate', 449), ('plate', 5534), ('truck', 13267)]
    # 数据集: shenzhentiaoqiao, [('bicyclist', 50), ('bus', 12175), ('car', 42484), ('fuzzy_plate', 26624), ('motorcyclist', 343), ('planted_plate', 2149), ('plate', 14239), ('tricycle', 3), ('truck', 3097)]
    # 数据集: anhuihuaibeigaosu, [('bus', 2783), ('car', 231859), ('fuzzy_plate', 90631), ('painted_plate', 448), ('plate', 17824), ('truck', 46680)]
    # 数据集: anhuihuaibeigaosu, XML_w_tank_van, [('bus', 2814), ('car', 217373), ('fuzzy_plate', 90140), ('painted_plate', 383), ('plate', 18424), ('tank', 7883), ('truck', 41283), ('van', 14174)]
    # 数据集: anhuihuaibeigaosu_night_diguangzhao, [('bus', 3629), ('car', 80339), ('fuzzy_plate', 21505), ('plate', 8605), ('truck', 11480)]
    # 数据集: paosa, [('bus', 326), ('car', 4846), ('fuzzy_plate', 6123), ('planted_plate', 15), ('plate', 1736), ('truck', 5190)]
    # 无车牌标注:
    # 数据集: shaobing（无车牌标注）, [('bicycle', 4510), ('bicyclist', 1587), ('bus', 74), ('car', 34502), ('motorcycle', 6232), ('motorcyclist', 16909), ('person', 15417), ('tank', 274), ('tricycle', 710), ('truck', 3518)]
    
    # 数据集: ZG_ZHJYZ_detection 加油站测试样本
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/"
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
    # AHHBAS_kakou1: [('car', 1218), ('fuzzy_plate', 545), ('plate', 92), ('truck', 223)]
    # AHHBAS_kakou2: [('car', 1991), ('fuzzy_plate', 1156), ('plate', 202), ('truck', 167)]
    # AHHBAS_kakou3: [('car', 3838), ('fuzzy_plate', 280), ('plate', 194), ('truck', 211)]
    # AHHBAS_kakou4: [('car', 2183), ('fuzzy_plate', 116), ('plate', 169), ('truck', 42)]
    # AHHBAS_kakou2_night: [('car', 290), ('fuzzy_plate', 91), ('plate', 37), ('truck', 88)]
    # AHHBAS_kakou3_night: [('car', 621), ('fuzzy_plate', 69), ('plate', 70), ('truck', 22)]
    # AHHBPS: [('bus', 12), ('car', 294), ('fuzzy_plate', 160), ('plate', 187), ('truck', 198)]
    # TXSDFX_6: [('bus', 3), ('car', 94), ('fuzzy_plate', 54), ('plate', 96), ('truck', 57)]
    # TXSDFX_7: [('car', 14), ('fuzzy_plate', 4), ('plate', 164), ('truck', 155)]
    # TXSDFX_9: [('car', 30), ('fuzzy_plate', 119), ('painted_plate', 113), ('plate', 2), ('truck', 135)]
    # TXSDFX_c: [('car', 34), ('fuzzy_plate', 4), ('plate', 157), ('truck', 138)]
    # SZTQ: [('bus', 226), ('car', 1414), ('fuzzy_plate', 431), ('painted_plate', 63), ('plate', 462), ('truck', 18)]

    # 数据集: ZG_BMX_detection 智观斑马线数据集
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/original/rongheng_night_hongwai/"
    # daminghu: [('bicycle', 5479), ('bicyclist', 8946), ('bus', 2911), ('car', 26364), ('head', 139209), ('helmet', 11468), ('motorcycle', 12987), ('motorcyclist', 18471), ('person', 110263), ('tricycle', 145), ('truck', 193)]
    # daminghu_night: [('bicycle', 1267), ('bicyclist', 654), ('bus', 159), ('car', 2112), ('head', 17237), ('helmet', 1046), ('motorcycle', 3795), ('motorcyclist', 3827), ('person', 16402), ('tricycle', 1108), ('truck', 18)]
    # shandongyingzikou: [('bus', 125), ('car', 7121), ('motorcycle', 524), ('motorcyclist', 442), ('person', 844), ('tricycle', 1240), ('truck', 4067)]
    # shandongyingzikou_night_hongwai: [('bus', 29), ('car', 1862), ('motorcyclist', 149), ('person', 18), ('tricycle', 135), ('truck', 852)]
    # yongzou_night_hongwai: [('bicycle', 4), ('bicyclist', 333), ('bus', 82), ('car', 2896), ('head', 1), ('motorcycle', 4786), ('motorcyclist', 13058), ('person', 35896), ('truck', 167)]
    # shenzhenlukou: [('bicyclist', 165), ('bus', 645), ('car', 15227), ('motorcyclist', 408), ('person', 1085), ('truck', 373)]
    # shenzhenlukou_night_hongwai: [('bicycle', 48), ('bicyclist', 609), ('bus', 604), ('car', 13744), ('motorcycle', 3), ('motorcyclist', 1320), ('person', 10584), ('truck', 291)]
    # shenzhenlukou_night_diguangzhao: [('bicyclist', 36), ('bus', 148), ('car', 10717), ('motorcyclist', 405), ('person', 930), ('truck', 269)]
    # rongheng: [('bicycle', 2240), ('bicyclist', 3214), ('bus', 1663), ('car', 40147), ('head', 40575), ('helmet', 3714), ('motorcycle', 1911), ('motorcyclist', 5577), ('person', 41595), ('tricycle', 21), ('truck', 675)]
    # rongheng_night_hongwai: [('bicycle', 1207), ('bicyclist', 2161), ('bus', 714), ('car', 36140), ('head', 36075), ('helmet', 2170), ('motorcycle', 2473), ('motorcyclist', 3984), ('person', 39228), ('tricycle', 2), ('truck', 343)]

    # 数据集: ZG_BMX_detection 智观斑马线数据集，非机动车裁剪
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/original_crop_moto/rongheng_night_hongwai/"
    # daminghu: [('head', 46623), ('helmet', 10424)]
    # daminghu_night: [('head', 7625), ('helmet', 1191)]
    # rongheng: [('head', 9846), ('helmet', 4103)]
    # rongheng_night_hongwai: [('head', 6675), ('helmet', 2359)]

    # 数据集: ZG_BMX_detection 斑马线测试样本
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/banmaxian_test_image/"
    # 2M_DaMingHu_far: [('bicycle', 119), ('bicyclist', 59), ('bus', 10), ('car', 93), ('head', 703), ('helmet', 100), ('motorcycle', 148), ('motorcyclist', 249), ('person', 556), ('tricycle', 5)]
    # 2M_DaMingHu_near: [('bicycle', 31), ('bicyclist', 31), ('bus', 4), ('car', 75), ('head', 494), ('helmet', 68), ('motorcycle', 171), ('motorcyclist', 154), ('person', 410), ('plate', 1), ('trcycle', 3), ('truck', 6)]
    # 2M_DaMingHu_night_far: [('bicycle', 71), ('bicyclist', 21), ('bus', 7), ('car', 95), ('head', 304), ('helmet', 38), ('motorcycle', 171), ('motorcyclist', 113), ('person', 297), ('trcycle', 4), ('truck', 2)]
    # 2M_DaMingHu_night_near: [('bicycle', 57), ('bicyclist', 12), ('bus', 1), ('car', 62), ('head', 303), ('helmet', 43), ('motorcycle', 104), ('motorcyclist', 101), ('person', 236), ('tricycle', 2), ('truck', 3)]
    # 2M_RongHeng_far: [('bicycle', 28), ('bicyclist', 34), ('bus', 48), ('car', 790), ('head', 303), ('helmet', 51), ('motorcycle', 3), ('motorcyclist', 96), ('person', 367), ('truck', 5)]
    # 2M_RongHeng_near: [('bicycle', 65), ('bicyclist', 15), ('bus', 1), ('car', 60), ('head', 299), ('helmet', 57), ('motorcycle', 2), ('motorcyclist', 72), ('person', 297), ('truck', 3)]
    # 2M_RongHeng_night_far: [('bicycle', 35), ('bicyclist', 26), ('bus', 21), ('car', 845), ('head', 559), ('helmet', 57), ('motorcycle', 17), ('motorcyclist', 122), ('person', 557), ('truck', 3)]
    # 2M_RongHeng_night_near: [('bicycle', 46), ('bicyclist', 5), ('bus', 2), ('car', 85), ('head', 253), ('helmet', 44), ('motorcycle', 59), ('motorcyclist', 66), ('person', 266)]

    # 开源数据集: MOT17\MOT20\HT21\NightOwls\Cityscapes\Safety_helmet\VOC2028
    # args.input_dir = "/yuanhuan/data/image/Open_Source/MOT/HT21/" 
    # MOT17: [('car_bus_truck', 6084), ('person', 86982)]
    # MOT20: [('person', 931571)]
    # HT21: [('head', 1129559)]
    # NightOwls: bicyclist、motorcyclist 标签与项目标签存在差异, 需要清洗标志
    # NightOwls: [('bicyclist', 3107), ('bicyclist_o', 377), ('motorcyclist', 389), ('motorcyclist_o', 22), ('person', 21876), ('person_o', 3121)]
    # Cityscapes: bicycle、motorcycle 标签存在差异, 需要清洗标志
    # Cityscapes: [('bicycle', 4047), ('bicyclist', 1906), ('bus', 483), ('car', 31822), ('caravan', 69), ('license plate', 6289), ('motorcycle', 634), ('motorcyclist', 254), ('person', 21413), ('trailer', 89), ('train', 194), ('truck', 582)]
    # Safety_helmet: [('head', 87040), ('helmet', 7502)]
    # VOC2028: [('head', 111495), ('helmet', 8967)]
    
    # Annotations_CarBusTruckBicyclistMotorcyclistPerson/Annotations_CarBusTruckBicyclistMotorcyclistPerson_filter
    # MOT17: [('car', 5430), ('car_o', 654), ('person', 79005), ('person_o', 7977)]
    # NightOwls: [('neg', 2481), ('person', 21876), ('person_o', 3121)]
    # Cityscapes: [('bicyclist', 1266), ('bicyclist_o', 640), ('bus', 572), ('bus_o', 105), ('car', 21190), ('car_o', 10701), ('motorcyclist', 164), ('motorcyclist_o', 90), ('neg', 3), ('person', 8113), ('person_o', 13300), ('truck', 530), ('truck_o', 141)]

    # 数据集: other
    # args.input_dir = "/yuanhuan/data/image/temp/Char_dec_dataset/"
    # Char_dec_dataset: [('char', 1900), ('char_ch', 300)]

    # 数据集: RM_upspeed
    # args.input_dir = "/yuanhuan/data/image/RM_upspeed/original/Europe/"
    # args.input_dir = "/yuanhuan/data/image/RM_upspeed/original/China/"
    # args.input_dir = "/yuanhuan/data/image/RM_upspeed/original/America_500w/"
    # args.input_dir = "/yuanhuan/data/image/RM_upspeed/original/America_200w/"
    # Europe: [('car', 3707), ('motorcyclist', 10), ('person', 22), ('trafficlight_green_n', 79), ('trafficlight_off', 5), ('trafficlight_yellow_n', 6), ('truck', 64), ('upspeed_spain_100', 189), ('upspeed_spain_120', 56), ('upspeed_spain_30', 28), ('upspeed_spain_40', 413), ('upspeed_spain_50', 2), ('upspeed_spain_60', 440), ('upspeed_spain_70', 85), ('upspeed_spain_80', 165), ('upspeed_spain_90', 90), ('zebra_crossing', 27)]
    # China: [('car', 782), ('person', 288), ('person_o', 33), ('sign_hand_c', 17), ('sign_handb_c', 5), ('sign_height_c', 10), ('sign_upspeed_c', 285), ('trafficlight', 304), ('zebra_crossing', 159)]
    # America_500w: [('downspeed_45', 46), ('upspeedneg_252', 43), ('upspeedtruck_65', 73), ('upspeedy_203', 58), ('upspeedy_25', 24), ('upspeedy_40', 64), ('upspeedy_45', 135), ('upspeedy_70', 156)]
    # America_200w: [('arrow_l', 10), ('arrow_r', 15), ('arrow_u', 32), ('arrow_ur', 2), ('car', 1010), ('person', 35), ('person_o', 4), ('sign_15', 16), ('sign_20', 24), ('sign_25', 11), ('sign_25_m', 2), ('sign_30', 25), ('sign_30_m', 2), ('sign_30_special', 4), ('sign_35', 21), ('sign_35_m', 1), ('sign_35_special', 2), ('sign_35_special_m', 1), ('sign_40', 22), ('sign_40_m', 4), ('sign_40_special', 4), ('sign_45', 21), ('sign_45_m', 1), ('sign_45_special', 1), ('sign_5', 10), ('sign_50', 10), ('sign_55', 18), ('sign_60', 21), ('sign_65', 15), ('sign_70', 7), ('sign_99_neg', 5), ('sign_stop', 237), ('trafficlight_2', 40), ('trafficlight_4_1_l_2_l', 2), ('trafficlight_4_1_x', 1), ('trafficlight_4_1_x_2_x', 2), ('trafficlight_5', 2), ('trafficlight_share0_green_l', 1), ('trafficlight_share0_green_n', 92), ('trafficlight_share0_green_u', 1), ('trafficlight_share0_green_x', 2), ('trafficlight_share0_off', 4), ('trafficlight_share0_red_l', 1), ('trafficlight_share0_red_n', 31), ('trafficlight_share0_red_x', 1), ('trafficlight_share0_yellow_n', 3), ('trafficlight_share1_green_l_green_n', 2), ('trafficlight_share1_off', 1), ('trafficlight_share1_off_green_n', 8), ('trafficlight_share1_off_yellow_n', 1), ('trafficlight_share1_red_n', 11), ('upspeed_10', 5), ('upspeed_15', 3), ('upspeed_20', 15), ('upspeed_25', 26), ('upspeed_30', 55), ('upspeed_35', 63), ('upspeed_40', 52), ('upspeed_45', 64), ('upspeed_5', 5), ('upspeed_50', 36), ('upspeed_55', 18), ('upspeed_60', 33), ('upspeed_65', 57), ('upspeed_70', 14), ('upspeedy_15', 10), ('upspeedy_20', 10), ('upspeedy_25', 17), ('upspeedy_30', 2), ('upspeedy_35', 2), ('upspeedy_40', 7), ('upspeedy_45', 1), ('zebra_crossing', 51)]

    # 数据集: Taxi_remnant
    # args.input_dir = "/yuanhuan/data/image/Taxi_remnant/original/shenzhen/20230616/"
    # args.input_dir = "/yuanhuan/data/image/Taxi_remnant/original/shenzhen/20230626/"
    # args.input_dir = "/yuanhuan/data/image/Taxi_remnant/original/shenzhen/20230628/"
    # args.input_dir = "/yuanhuan/data/image/Taxi_remnant/original/shenzhen/20230702/"
    # args.input_dir = "/yuanhuan/data/image/Taxi_remnant/original/shenzhen/20230703/"
    # args.input_dir = "/yuanhuan/data/image/Taxi_remnant/original/shenzhen/20230704/"
    # args.input_dir = "/yuanhuan/data/image/Taxi_remnant/original/shenzhen/BYDe6_middle_20230720/"
    # args.input_dir = "/yuanhuan/data/image/Taxi_remnant/original/shenzhen/BYDe6_side_20230720/"
    # args.input_dir = "/yuanhuan/data/image/Taxi_remnant/original/shenzhen/Camry_middle_20230719/"
    # args.input_dir = "/yuanhuan/data/image/Taxi_remnant/original/shenzhen/Camry_middle_20230720/"
    # args.input_dir = "/yuanhuan/data/image/Taxi_remnant/original/shenzhen/Camry_side_20230719/"
    # args.input_dir = "/yuanhuan/data/image/Taxi_remnant/original/shenzhen/MKZ_middle_20230721/"
    # args.input_dir = "/yuanhuan/data/image/Taxi_remnant/original/shenzhen/MKZ_side_20230721/"
    args.input_dir = "/yuanhuan/data/image/Taxi_remnant/original/shenzhen/Pickup_middle_20230721/"
    # shenzhen/20230616: [('remnants', 271)]
    # shenzhen/20230626: [('remnants', 1191)]
    # shenzhen/20230628: [('remnants', 10237)]
    # shenzhen/20230702: [('remnants', 7317)]
    # shenzhen/20230703]: [('remnants', 7241)]
    # shenzhen/20230704: [('remnants', 14123)]
    # shenzhen/BYDe6_middle_20230720: [('remnants', 8140)]
    # shenzhen/BYDe6_side_20230720: [('remnants', 6899)]
    # shenzhen/Camry_middle_20230719: [('remnants', 6155)]
    # shenzhen/Camry_middle_20230720: [('remnants', 4131)]
    # shenzhen/Camry_side_20230719: [('remnants', 9024)]
    # shenzhen/MKZ_middle_20230721: [('remnants', 7187)]
    # shenzhen/MKZ_side_20230721: [('remnants', 5385)]
    # shenzhen/Pickup_middle_20230721: [('remnants', 11198)]

    args.trainval_file = args.input_dir + "ImageSets/Main/trainval.txt"
    args.train_file = args.input_dir + "ImageSets/Main/train.txt"
    args.val_file = args.input_dir + "ImageSets/Main/val.txt"
    args.test_file = args.input_dir + "ImageSets/Main/test.txt"
    args.statistic_dict = {'trainval': args.trainval_file, 'test': args.test_file }

    # args.jpg_dir =  args.input_dir + "JPEGImages/"
    args.jpg_dir =  args.input_dir + "image/"
    # args.xml_dir =  args.input_dir + "Annotations/"
    args.xml_dir =  args.input_dir + "xml/"
    # args.xml_dir =  args.input_dir + "refine_xml/"
    # args.xml_dir =  args.input_dir + "Annotations_CarBusTruckMotorcyclePlateMotoplate_w_fuzzy/"
    # args.xml_dir =  args.input_dir + "Annotations_CarBusTruckMotorcyclePlate_MMDetect/"
    # args.xml_dir =  args.input_dir + "Annotations_License/"

    statistic_classname(args)
    # statistic_classname_train_val_test(args)