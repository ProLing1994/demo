import numpy as np
import xml.etree.ElementTree as ET

kind_num_labels = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
                    "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
                    "陕", "甘", "青", "宁", "新", "警", "学", '挂', '领', '空', '港', '澳', 
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 
                    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']

status_name_list = ['n', 'f', 'o']              # n 表示车牌能看清, f 表示车牌因反光或太小等因素模糊而不能识别文字, o 表示车牌遮挡车牌不全的情况
column_name_list = ['none', 'double', 'single']
color_name_list = ['none', 'red', 'green', 'yellow', 'blue', 'white', 'infrared', 'black']

# 无用标签修正
replace_name_list = ['.', '#', ' ', '　', '"', '”']
# 错误标注修正
replace_name_dict = {
                        'O': '0', 
                        'I': '1',
                        '泸': '沪',
                        '贛': '赣',
                        '翼': '冀',
                        '闵': '闽',
                        '峡': '陕',
                        '晥': '皖',
                        'Ａ': 'A',
                        'Ｂ': 'B',
                        'Ｃ': 'C',
                        'Ｄ': 'D',
                        'Ｅ': 'E',
                        'Ｆ': 'F',
                        'Ｇ': 'G',
                        'Ｈ': 'H',
                        'Ｊ': 'J',
                        'Ｋ': 'K',
                        'Ｌ': 'L',
                        'Ｍ': 'M',
                        'Ｎ': 'N',
                        '０': '0',
                        'Ｑ': 'Q',
                        'Ｓ': 'S',
                        'Ｔ': 'T',
                        'Ｕ': 'U',
                        'Ｗ': 'W',
                        'Ｙ': 'Y',
                        '８': '8',
                    }

analysis_label_columns = {
                            'num': kind_num_labels,
                            'column': column_name_list,
                            'color': color_name_list,
                        }

# method 
def get_color(idx):
    idx = idx * 5
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def json_load_object_plate_points(cell):
    
    # plate points
    if 'type' in cell and cell["type"] == "rectangle":
        pts = np.array(cell["points"], np.int32)
        pts = pts.reshape((-1, 1, 2))

        x1 = np.min((pts[0][0][0], pts[1][0][0]))
        x2 = np.max((pts[0][0][0], pts[1][0][0]))
        y1 = np.min((pts[0][0][1], pts[1][0][1]))
        y2 = np.max((pts[0][0][1], pts[1][0][1]))
        w = x2 - x1
        h = y2 - y1
    
    else:
        pts = np.array(cell["points"], np.int32)
        pts = pts.reshape((-1, 1, 2))

        x1 = np.min((pts[0][0][0], pts[1][0][0], pts[2][0][0], pts[3][0][0]))
        x2 = np.max((pts[0][0][0], pts[1][0][0], pts[2][0][0], pts[3][0][0]))
        y1 = np.min((pts[0][0][1], pts[1][0][1], pts[2][0][1], pts[3][0][1]))
        y2 = np.max((pts[0][0][1], pts[1][0][1], pts[2][0][1], pts[3][0][1]))
        w = x2 - x1
        h = y2 - y1

    return x1, x2, y1, y2, w, h


def json_load_object_plate_status(cell):
    
    # plate status
    plate_status = status_name_list[0]

    if 'attributes' in cell and len(cell["attributes"]):
        for json_attributes in cell["attributes"]:
            if json_attributes["name"] == "status":
                plate_status = json_attributes["value"]

    return plate_status


def json_load_object_plate_color(cell):

    # plate color
    plate_color = color_name_list[0]
    load_plate_color = color_name_list[0]

    if 'vidcolor' in cell:
        load_plate_color = cell['vidcolor']

        if load_plate_color == 'red_card':
            plate_color = color_name_list[1]
        elif load_plate_color == 'green_card':
            plate_color = color_name_list[2]
        elif load_plate_color == 'yellow_card':
            plate_color = color_name_list[3]
        elif load_plate_color == 'blue_card':
            plate_color = color_name_list[4]
        elif load_plate_color == 'white_card':
            plate_color = color_name_list[5]
        elif load_plate_color == 'infrared_card':
            plate_color = color_name_list[6]
        elif load_plate_color == 'black_card':
            plate_color = color_name_list[7]

    elif 'attributes' in cell and len(cell["attributes"]):
        for json_attributes in cell["attributes"]:
            if json_attributes["name"] == "color":
                load_plate_color = json_attributes["value"]
        
        if load_plate_color == 'red':
            plate_color = color_name_list[1]
        elif load_plate_color == 'green':
            plate_color = color_name_list[2]
        elif load_plate_color == 'yellow':
            plate_color = color_name_list[3]
        elif load_plate_color == 'blue':
            plate_color = color_name_list[4]
        elif load_plate_color == 'white':
            plate_color = color_name_list[5]
        elif load_plate_color == 'infrared':
            plate_color = color_name_list[6]
        elif load_plate_color == 'black':
            plate_color = color_name_list[7]
    
    return plate_color, load_plate_color


def json_load_object_plate_column(cell, w, h):

    # plate column
    plate_column = column_name_list[0]
    load_plate_column = column_name_list[0]
    
    if 'vidlan' in cell:
        load_plate_column = cell['vidlan']

        if load_plate_column == 'Double_column':
            plate_column = column_name_list[1]
        elif load_plate_column == 'Single_column':
            plate_column = column_name_list[2]

    elif 'attributes' in cell and len(cell["attributes"]):
        for json_attributes in cell["attributes"]:
            if json_attributes["name"] == "column":
                load_plate_column = json_attributes["value"]

    if plate_column == column_name_list[0]:
        if( w / h > 2.5 ):
            plate_column = column_name_list[2]
        else:
            plate_column = column_name_list[1]
    
    return plate_column, load_plate_column


def json_load_object_plate_num(cell):

    # plate num
    plate_num = 'none'
    if 'vehicleic' in cell:
        plate_num = cell["vehicleic"]
    elif 'attributes' in cell and len(cell["attributes"]):
        for json_attributes in cell["attributes"]:
            if json_attributes["name"] == "number":
                plate_num = json_attributes["value"]
    else:
        pass
    
    if plate_num != 'none':
        plate_num = plate_num.upper()
        for replace_name in replace_name_list:
            plate_num = plate_num.replace(replace_name, '')
        
        for replace_name in replace_name_dict.keys():
            plate_num = plate_num.replace(replace_name, replace_name_dict[replace_name])

    return plate_num