import io
import json
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/')
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

                # city
                'ad': 3,
                'abudhabi': 3,
                'dubai': 4,
                'ajman': 5,
                'sharjah': 6,
                'shj': 6,
                'rak': 7,
                'ummalqaiwain': 8,
                'fujairah': 9,

                # car_type
                'taxi': 10,
                'police': 11,
                'trp': 12,
                'ptr': 13,
                'trade': 14,

                # kind
                'kind': 15,

                # num
                'num': 16,
                }


name_2_mask_id_dict = {
                # background [0, 0, 0]

                # country
                'uae': [1, 1, 1],               
                'oman': [2, 2, 2],             

                # city
                'ad': [3, 3, 3],
                'abudhabi': [3, 3, 3],
                'dubai': [4, 4, 4],
                'ajman': [5, 5, 5],
                'sharjah': [6, 6, 6],
                'shj': [6, 6, 6],
                'rak': [7, 7, 7],
                'ummalqaiwain': [8, 8, 8],
                'fujairah': [9, 9, 9],
                
                # car_type
                'taxi': [10, 10, 10],
                'police': [11, 11, 11],
                'trp': [12, 12, 12],
                'ptr': [13, 13, 13],
                'trade': [14, 14, 14],

                # kind
                'kind': [15, 15, 15],

                # num
                'num': [16, 16, 16],
                }


name_2_mask_color_dict = {
                # background [0, 0, 0]

                # country
                'uae': get_color(1),
                'oman': get_color(2),

                # city
                'ad': get_color(3),
                'abudhabi': get_color(3),
                'dubai': get_color(4),
                'ajman': get_color(5),
                'sharjah': get_color(6),
                'shj':  get_color(6),
                'rak': get_color(7),
                'ummalqaiwain': get_color(8),
                'fujairah': get_color(9),

                # car_type
                'taxi': get_color(10),
                'police': get_color(11),
                'trp': get_color(12),
                'ptr': get_color(13),
                'trade': get_color(14),

                # kind
                'kind': get_color(15),

                # num
                'num': get_color(16),
                }


id_2_mask_name_dict = {
                # country
                1: 'uae',
                2: 'oman',

                # city
                3: 'abudhabi',
                4: 'dubai',
                5: 'ajman',
                6: 'sharjah',
                7: 'rak',
                8: 'ummalqaiwain',
                9: 'fujairah',
                
                # car_type
                10: 'taxi',
                11: 'police',
                12: 'trp',
                13: 'ptr',
                14: 'trade',

                # kind
                15: 'kind',

                # num
                16: 'num',
                }


id_2_mask_id_dict = {
                # country
                1: [1, 1, 1],
                2: [2, 2, 2],

                # city                
                3: [3, 3, 3],
                4: [4, 4, 4],
                5: [5, 5, 5],
                6: [6, 6, 6],
                7: [7, 7, 7],
                8: [8, 8, 8],
                9: [9, 9, 9],
                
                # car_type
                10: [10, 10, 10],
                11: [11, 11, 11],
                12: [12, 12, 12],
                13: [13, 13, 13],
                14: [14, 14, 14],

                # kind
                15: [15, 15, 15],

                # num
                16: [16, 16, 16],
                }


id_2_mask_color_dict = {
                # country
                1: get_color(1),
                2: get_color(2),

                # city
                3: get_color(3),
                4: get_color(4),
                5: get_color(5),
                6: get_color(6),
                7: get_color(7),
                8: get_color(8),
                9: get_color(9),

                # car_type
                10: get_color(10),
                11: get_color(11),
                12: get_color(12),
                13: get_color(13),
                14: get_color(14),
                
                # kind
                15: get_color(15),

                # num
                16: get_color(16),
                }


class_seg_label = ['bkg','uae','oman','abudhabi','dubai','ajman','sharjah','rak','ummalqaiwain','fujairah','taxi','police','trp','ptr','trade', 'kind','num']
# class_seg_weight = [0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5]
class_seg_weight = [0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5, 0.5]
add_mask_order = ['num','kind','uae','oman','ad','abudhabi','dubai','ajman','sharjah','shj','rak','ummalqaiwain','fujairah','taxi','police','trp','ptr','trade']

country_mask_id_list = [1, 2]
city_mask_id_list = [3, 4, 5, 6, 7, 8, 9]
car_type_mask_id_list = [10, 11, 12, 13, 14]
kind_mask_id_list = [15]
num_mask_id_list = [16]

country_mask_name_list = ['uae','oman']
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
