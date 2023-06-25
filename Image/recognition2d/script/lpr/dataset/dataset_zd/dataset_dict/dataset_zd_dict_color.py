import io
import json
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/')
from Image.recognition2d.script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_normal import *

name_2_id_dict = {   
                # background
                'none': 0,

                # color
                'red': 1,
                'green': 2,
                'yellow': 3,
                }


name_2_mask_id_dict = {
                # background [0, 0, 0]

                # color
                'red': [1, 1, 1], 
                'green': [2, 2, 2], 
                'yellow': [3, 3, 3], 
                }


name_2_mask_color_dict = {
                # background [0, 0, 0]

                # color
                'red': get_color(20),
                'green': get_color(21),
                'yellow': get_color(22),
                }

id_2_mask_name_dict = {
                # color
                1: 'red',
                2: 'green',
                3: 'yellow',
                }

id_2_mask_id_dict = {
                # color
                1: [1, 1, 1],       
                2: [2, 2, 2],
                3: [3, 3, 3],
                }


id_2_mask_color_dict = {
                # color
                1: get_color(20),
                2: get_color(21),
                3: get_color(22),
                }


class_seg_label = ['bkg','red','green','yellow']
class_seg_weight = [0.1, 1, 1, 1]
add_mask_order = ['red','green','yellow']

color_mask_id_list = [1, 2, 3]
color_mask_name_list = ['red','green','yellow']

class_seg_label_group = ['color']
class_seg_label_group_threh_map = {
                                'color': 500, 
                            }
class_seg_label_group_2_id_map = {
                                'color': color_mask_id_list, 
                            }
class_seg_label_group_2_name_map = {
                                'color': color_mask_name_list, 
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
            if classname in country_name_list:
                continue
            if classname in country_f_name_list:
                continue
            if classname in city_name_list:
                continue
            if classname in city_f_name_list:
                continue
            if classname in car_type_name_list:
                continue
            if classname in car_type_f_name_list:
                continue
            if classname in kind_name_list:
                continue
            if classname in num_name_list:
                continue

            if classname not in name_2_id_dict or \
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
            if classname in country_name_list:
                continue
            if classname in country_f_name_list:
                continue
            if classname in city_name_list:
                continue
            if classname in city_f_name_list:
                continue
            if classname in car_type_name_list:
                continue
            if classname in car_type_f_name_list:
                continue
            if classname in kind_name_list:
                continue
            if classname in num_name_list:
                continue

            if classname not in name_2_id_dict or \
                classname not in name_2_mask_color_dict:
                print(classname)
                continue
            
            object_roi_list.append({"classname": classname, "bndbox":bndbox})

    return object_roi_list