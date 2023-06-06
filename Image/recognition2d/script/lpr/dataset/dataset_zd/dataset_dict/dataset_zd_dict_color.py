import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
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