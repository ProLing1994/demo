import io
import json
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, '/home/huanyuan/code/demo/')
# sys.path.insert(0, '/yuanhuan/code/demo/')
from Image.recognition2d.script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_normal import *

##############################################
# TODO（临时方案，city_f_2_city）
change_country_f_2_country_dict = {
                        'oman_f': 'oman', 
                    }
change_city_f_2_city_dict = {
                        'ummalqaiwain_f': 'ummalqaiwain', 
                        'fujairah_f': 'fujairah'
                    }
change_car_type_f_2_car_type_dict = {
                        'trade_f': 'trade', 
                    }

name_2_mask_name_dict = {   
                # background
                'none': 'none',

                # country
                'uae': 'uae',
                'oman': 'oman',
                'ksa': 'ksa',

                # city
                'ad': 'abudhabi',
                'abudhabi': 'abudhabi',
                'dubai': 'dubai',
                'ajman': 'ajman',
                'sharjah': 'sharjah',
                'shj': 'sharjah',
                'rak': 'rak',
                'ummalqaiwain': 'ummalqaiwain',
                'fujairah': 'fujairah',

                # car_type
                'taxi': 'taxi',
                'police': 'police',
                'trp': 'trp',
                'ptr': 'ptr',
                'trade': 'trade',

                # kind
                'kind': 'kind',

                # num
                'num': 'num',
                }


name_2_id_dict = {   
                # background
                'none': 0,

                # country
                'uae': 1,
                'oman': 2,
                'ksa': 3,

                # city
                'ad': 4,
                'abudhabi': 4,
                'dubai': 5,
                'ajman': 6,
                'sharjah': 7,
                'shj': 7,
                'rak': 8,
                'ummalqaiwain': 9,
                'fujairah': 10,

                # car_type
                'taxi': 11,
                'police': 12,
                'trp': 13,
                'ptr': 14,
                'trade': 15,

                # kind
                'kind': 16,

                # num
                'num': 17,
                }


name_2_mask_id_dict = {
                # background [0, 0, 0]

                # country
                'uae': [1, 1, 1],  
                'oman': [2, 2, 2],
                'ksa': [3, 3, 3],

                # city
                'ad': [4, 4, 4],
                'abudhabi': [4, 4, 4],
                'dubai': [5, 5, 5],
                'ajman': [6, 6, 6],
                'sharjah': [7, 7, 7],
                'shj': [7, 7, 7],
                'rak': [8, 8, 8],
                'ummalqaiwain': [9, 9, 9],
                'fujairah': [10, 10, 10],
                
                # car_type
                'taxi': [11, 11, 11],
                'police': [12, 12, 12],
                'trp': [13, 13, 13],
                'ptr': [14, 14, 14],
                'trade': [15, 15, 15],

                # kind
                'kind': [16, 16, 16],

                # num
                'num': [17, 17, 17],
                }


name_2_mask_color_dict = {
                # background [0, 0, 0]

                # country
                'uae': get_color(1),
                'oman': get_color(2),
                'ksa': get_color(3),

                # city
                'ad': get_color(4),
                'abudhabi': get_color(4),
                'dubai': get_color(5),
                'ajman': get_color(6),
                'sharjah': get_color(7),
                'shj':  get_color(7),
                'rak': get_color(8),
                'ummalqaiwain': get_color(9),
                'fujairah': get_color(10),

                # car_type
                'taxi': get_color(11),
                'police': get_color(12),
                'trp': get_color(13),
                'ptr': get_color(14),
                'trade': get_color(15),

                # kind
                'kind': get_color(16),

                # num
                'num': get_color(17),
                }


id_2_mask_name_dict = {
                # country
                1: 'uae',
                2: 'oman',
                3: 'ksa',

                # city
                4: 'abudhabi',
                5: 'dubai',
                6: 'ajman',
                7: 'sharjah',
                8: 'rak',
                9: 'ummalqaiwain',
                10: 'fujairah',
                
                # car_type
                11: 'taxi',
                12: 'police',
                13: 'trp',
                14: 'ptr',
                15: 'trade',

                # kind
                16: 'kind',

                # num
                17: 'num',
                }


id_2_mask_id_dict = {
                # country
                1: [1, 1, 1],
                2: [2, 2, 2],
                3: [3, 3, 3],

                # city
                4: [4, 4, 4],
                5: [5, 5, 5],
                6: [6, 6, 6],
                7: [7, 7, 7],
                8: [8, 8, 8],
                9: [9, 9, 9],
                10: [10, 10, 10],
                
                # car_type
                11: [11, 11, 11],
                12: [12, 12, 12],
                13: [13, 13, 13],
                14: [14, 14, 14],
                15: [15, 15, 15],

                # kind
                16: [16, 16, 16],

                # num
                17: [17, 17, 17],
                }


id_2_mask_color_dict = {
                # country
                1: get_color(1),
                2: get_color(2),
                3: get_color(3),

                # city
                4: get_color(4),
                5: get_color(5),
                6: get_color(6),
                7: get_color(7),
                8: get_color(8),
                9: get_color(9),
                10: get_color(10),

                # car_type
                11: get_color(11),
                12: get_color(12),
                13: get_color(13),
                14: get_color(14),
                15: get_color(15),
                
                # kind
                16: get_color(16),

                # num
                17: get_color(17),
                }


class_seg_label = ['bkg','uae','oman','ksa','abudhabi','dubai','ajman','sharjah','rak','ummalqaiwain','fujairah','taxi','police','trp','ptr','trade', 'kind','num']
class_seg_weight = [0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
add_mask_order = ['num','kind','uae','oman','ksa','ad','abudhabi','dubai','ajman','sharjah','shj','rak','ummalqaiwain','fujairah','taxi','police','trp','ptr','trade']

country_mask_id_list = [1, 2, 3]
city_mask_id_list = [4, 5, 6, 7, 8, 9, 10]
car_type_mask_id_list = [11, 12, 13, 14, 15]
kind_mask_id_list = [16]
num_mask_id_list = [17]

country_mask_name_list = ['uae','oman','ksa']
city_mask_name_list = ['abudhabi','dubai','ajman','sharjah','rak','ummalqaiwain','fujairah']
car_type_mask_name_list = ['taxi','police','trp','ptr','trade']   
kind_mask_name_list = ['kind']
num_mask_name_list = ['num']

class_seg_label_group = ['country', 'city', 'car_type', 'kind', 'num']
class_seg_label_group_threh_map = {
                                'country': 80, 
                                'city': 10, 
                                'car_type': 80,
                                'kind': 80,
                                'num': 820,
                            }
class_seg_label_group_2_id_map = {
                                'country': country_mask_id_list, 
                                'city': city_mask_id_list, 
                                'car_type': car_type_mask_id_list,
                                'kind': kind_mask_id_list,
                                'num': num_mask_id_list,
                            }

class_seg_label_group_2_name_map = {
                                'country': country_mask_name_list, 
                                'city': city_mask_name_list, 
                                'car_type': car_type_mask_name_list,
                                'kind': kind_mask_name_list,
                                'num': num_mask_name_list,
                            }


# method
def load_object_roi(xml_path, json_path, new_style):
    
    object_roi_list = []

    if not new_style:
        if not os.path.exists(xml_path):
            return object_roi_list
            
        # xml
        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

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
        
            classname = classname.lower()

            if classname in replace_name_dict:
                classname = replace_name_dict[classname]        

            if classname in ignore_unknown_label:
                continue

            #################################
            # （TODO）以下类别，暂时不参与训练
            #################################
            if classname in change_country_f_2_country_dict:
                classname = change_country_f_2_country_dict[classname]
            if classname in change_city_f_2_city_dict:
                classname = change_city_f_2_city_dict[classname]
            if classname in change_car_type_f_2_car_type_dict:
                classname = change_car_type_f_2_car_type_dict[classname]
            if classname in country_f_name_list:
                continue
            if classname in city_f_name_list:
                continue
            if classname in car_type_f_name_list:
                continue
            if classname in color_name_list:
                continue
            if classname in kind_num_labels:
                continue

            if classname not in name_2_mask_id_dict or \
                classname not in name_2_mask_color_dict:
                print(classname)
                continue
            
            object_roi_list.append({"classname": classname, "bndbox":bndbox})
    
    else:
        if not os.path.exists(json_path):
            return object_roi_list
            
        # json
        with io.open(json_path, "r", encoding="UTF-8") as f:
            data_json = json.load(f, encoding='utf-8')
            f.close()

        for cell in data_json['shapes']:
            
            if cell["label"] == "license_kind" or \
                cell["label"] ==  "license_num":

                # name
                classname = str(cell["label"]).replace("license_", "")

            elif cell["label"] == "license_country" or \
                cell["label"] == "license_country_f" or \
                cell["label"] == "license_city" or \
                cell["label"] == "license_city_f" or \
                cell["label"] == "license_car_type" or \
                cell["label"] == "license_car_type_f" or \
                cell["label"] == "license_color" :

                # name
                classname = str(cell["attributes"][0]["value"])

            # bbox
            pts = np.array(cell["points"], np.int32)
            pts = pts.reshape((-1, 1, 2))

            x1 = np.min((pts[0][0][0], pts[1][0][0]))
            x2 = np.max((pts[0][0][0], pts[1][0][0]))
            y1 = np.min((pts[0][0][1], pts[1][0][1]))
            y2 = np.max((pts[0][0][1], pts[1][0][1]))

            # bbox
            bndbox = [x1, y1, x2, y2]

            if classname in ignore_unknown_label:
                continue
            #################################
            # （TODO）以下类别，暂时不参与训练
            #################################
            if classname in change_country_f_2_country_dict:
                classname = change_country_f_2_country_dict[classname]
            if classname in change_city_f_2_city_dict:
                classname = change_city_f_2_city_dict[classname]
            if classname in change_car_type_f_2_car_type_dict:
                classname = change_car_type_f_2_car_type_dict[classname]
            if classname in country_f_name_list:
                continue
            if classname in city_f_name_list:
                continue
            if classname in car_type_f_name_list:
                continue
            if classname in color_name_list:
                continue
            if classname in kind_num_labels:
                continue

            if classname not in name_2_mask_id_dict or \
                classname not in name_2_mask_color_dict:
                print(classname)
                continue
            
            object_roi_list.append({"classname": classname, "bndbox":bndbox})

    return object_roi_list