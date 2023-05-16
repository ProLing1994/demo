import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from Image.recognition2d.script.lpr.dataset.dataset_cn.dataset_dict.dataset_cn_dict_normal import *


name_2_id_dict = {   
                # background
                'none': 0,

                # color
                'green': 1,
                'yellow': 2,
                'blue': 3,
                'white': 4,
                'black': 5,
                }


name_2_mask_id_dict = {
                # background [0, 0, 0]

                # color
                'green': [1, 1, 1], 
                'yellow': [2, 2, 2], 
                'blue': [3, 3, 3], 
                'white': [4, 4, 4], 
                'black': [5, 5, 5], 
                }


name_2_mask_color_dict = {
                # background [0, 0, 0]

                # color
                'green': get_color(1),
                'yellow': get_color(2),
                'blue': get_color(3),
                'white': get_color(4),
                'black': get_color(5),
                }

id_2_mask_name_dict = {
                # color
                1: 'green',
                2: 'yellow',
                3: 'blue',
                4: 'white',
                5: 'black',
                }

id_2_mask_id_dict = {
                # color
                1: [1, 1, 1],       
                2: [2, 2, 2],
                3: [3, 3, 3],
                4: [4, 4, 4], 
                5: [5, 5, 5], 
                }


id_2_mask_color_dict = {
                # color
                1: get_color(1),
                2: get_color(2),
                3: get_color(3),
                4: get_color(4),
                5: get_color(5),
                }


class_seg_label = ['none', 'green', 'yellow', 'blue', 'white', 'black']
class_seg_weight = [0.1, 1, 1, 1, 1, 1]
add_mask_order = ['green', 'yellow', 'blue', 'white', 'black']


color_mask_id_list = [1, 2, 3, 4, 5]
color_mask_name_list = ['green', 'yellow', 'blue', 'white', 'black']

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
