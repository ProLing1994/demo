import argparse
import io
import os
import pandas as pd
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET
import shutil

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Basic.script.json.platform_json_write import PlatformJsonWriter

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from Image.recognition2d.script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_normal import *


def dataset_csv(args):

    # mkdir
    create_folder(args.output_csv_dir)
    create_folder(args.output_error_crop_data_dir)

    # init 
    csv_list = []           # [{"img_path": "", "json_path": "", "id": "", "name": "", "roi": "", "country": "", "city": "", "color": "", "column": "", "num": "", "crop_img": "", "crop_xml": ""}]
    error_list = []         # [{"img_path": "", "json_path": "", "crop_img": "", "crop_xml": "", "type": "", "value": ""}]

    # pd
    data_pd = pd.read_csv(args.input_csv_path) 

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        # info
        img_path = row['img_path']
        json_path = row['json_path']
        plate_id = row['id']
        plate_name = row['name']
        plate_roi = row['roi'] 
        plate_country = row['country'] 
        plate_city = row['city'] 
        plate_color = row['color'] 
        plate_column = row['column'] 
        plate_num = row['num'] 
        crop_img_path = os.path.join(args.img_dir, plate_name + ".jpg")
        crop_xml_path = os.path.join(args.xml_dir, plate_name + ".xml")

        if not args.new_style:

            # init 
            country_list = []
            city_list = []
            car_type_list = []
            country_f_list = []
            city_f_list = []
            car_type_f_list = []
            color_list = []

            # xml
            try:
                tree = ET.parse(crop_xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
                root = tree.getroot()   # 获取根节点
            except:
                print('"img_path": {}, "json_path": {}, "crop_img": {}, "crop_xml": {}, "type": "xml"'.format(img_path, json_path, crop_img_path, crop_xml_path))
                error_list.append({"img_path": img_path, "json_path": json_path, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "type": "xml", "value": ""})
                continue

            for object in root.findall('object'):
                # name
                classname = str(object.find('name').text)

                classname = classname.lower()

                if classname in replace_name_dict:
                    classname = replace_name_dict[classname]        

                if classname in ignore_unknown_label:
                    continue

                if classname in kind_name_list or \
                    classname in num_name_list or \
                    classname in country_name_list or \
                    classname in city_name_list or \
                    classname in car_type_name_list or \
                    classname in country_f_name_list or \
                    classname in city_f_name_list or \
                    classname in car_type_f_name_list or \
                    classname in color_name_list:

                    if classname in country_name_list:
                        country_list.append(classname) 
                    
                    if classname in city_name_list:
                        city_list.append(classname)

                    if classname in car_type_name_list:
                        car_type_list.append(classname)

                    if classname in country_f_name_list:
                        country_f_list.append(classname)

                    if classname in city_f_name_list:
                        city_f_list.append(classname)

                    if classname in car_type_f_name_list:
                        car_type_f_list.append(classname)

                    if classname in color_name_list:
                        color_list.append(classname)
                else:
                    print(classname)
                    raise Exception

            country_list = list(set(country_list))
            city_list = list(set(city_list))
            car_type_list = list(set(car_type_list))

            country_f_list = list(set(country_f_list))
            city_f_list = list(set(city_f_list))
            car_type_f_list = list(set(car_type_f_list))
            color_list = list(set(color_list))

            if len(country_list) > 1:
                # 标签存在问题，多个国家
                print('"img_path": {}, "json_path": {}, "crop_img": {}, "crop_xml": {}, "type": "country"'.format(img_path, json_path, crop_img_path, crop_xml_path))
                error_list.append({"img_path": img_path, "json_path": json_path, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "type": "country", "value": ""})
            else:
                pass

            if len(city_list) > 1:
                # 标签存在问题，多个城市
                print('"img_path": {}, "json_path": {}, "crop_img": {}, "crop_xml": {}, "type": "city"'.format(img_path, json_path, crop_img_path, crop_xml_path))
                error_list.append({"img_path": img_path, "json_path": json_path, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "type": "city", "value": ""})
            else:
                pass

            if len(car_type_list) > 1:
                # 标签存在问题，多个车型
                print('"img_path": {}, "json_path": {}, "crop_img": {}, "crop_xml": {}, "type": "car_type"'.format(img_path, json_path, crop_img_path, crop_xml_path))
                error_list.append({"img_path": img_path, "json_path": json_path, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "type": "car_type", "value": ""})
            else:
                pass

            if len(country_f_list) > 1:
                # 标签存在问题，多个国家
                print('"img_path": {}, "json_path": {}, "crop_img": {}, "crop_xml": {}, "type": "country_f"'.format(img_path, json_path, crop_img_path, crop_xml_path))
                error_list.append({"img_path": img_path, "json_path": json_path, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "type": "country_f", "value": ""})
            else:
                pass

            if len(city_f_list) > 1:
                # 标签存在问题，多个城市
                print('"img_path": {}, "json_path": {}, "crop_img": {}, "crop_xml": {}, "type": "city_f"'.format(img_path, json_path, crop_img_path, crop_xml_path))
                error_list.append({"img_path": img_path, "json_path": json_path, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "type": "city_f", "value": ""})
            else:
                pass

            if len(car_type_f_list) > 1:
                # 标签存在问题，多个车型
                print('"img_path": {}, "json_path": {}, "crop_img": {}, "crop_xml": {}, "type": "car_type_f"'.format(img_path, json_path, crop_img_path, crop_xml_path))
                error_list.append({"img_path": img_path, "json_path": json_path, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "type": "car_type_f", "value": ""})
            else:
                pass

            if len(color_list) > 1:
                # 标签存在问题，多个颜色
                print('"img_path": {}, "json_path": {}, "crop_img": {}, "crop_xml": {}, "type": "color"'.format(img_path, json_path, crop_img_path, crop_xml_path))
                error_list.append({"img_path": img_path, "json_path": json_path, "crop_img": crop_img_path, "crop_xml": crop_xml_path, "type": "color", "value": ""})
            else:
                pass

            csv_list.append({"img_path": img_path, "json_path": json_path, "id": plate_id, "name": plate_name, "roi": plate_roi, "country": plate_country, "city": plate_city, "color": plate_color, "column": plate_column, "num": plate_num, "crop_img": crop_img_path, "crop_xml": crop_xml_path})

    # out csv
    csv_pd = pd.DataFrame(csv_list)
    csv_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")

    error_data_csv_path = os.path.join(args.output_error_crop_data_dir, 'error.csv')
    error_pd = pd.DataFrame(error_list)
    error_pd.to_csv(error_data_csv_path, index=False, encoding="utf_8_sig")


def write_error_data(args):
    
    if not args.bool_write_error_data:
        return

    create_folder(args.output_error_data_img_dir)
    create_folder(args.output_error_data_xml_dir)

    # pd
    error_data_csv_path = os.path.join(args.output_error_crop_data_dir, 'error.csv')

    try:
        data_pd = pd.read_csv(error_data_csv_path)
    except:
        return
    
    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        # info
        crop_img_path = row['crop_img']
        crop_xml_path = row['crop_xml']
        img_name = os.path.basename(crop_img_path)
        xml_name = os.path.basename(crop_xml_path)

        # img
        try:
            to_img_path = os.path.join(args.output_error_data_img_dir, img_name)
            shutil.copy(crop_img_path, to_img_path)
        except:
            continue
        
        # xml
        try:
            to_xml_path = os.path.join(args.output_error_data_xml_dir, xml_name)
            shutil.copy(crop_xml_path, to_xml_path)
        except:
            continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="uae_2022_city_RAK_0") 
    parser.add_argument('--input_csv_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_csv/") 
    parser.add_argument('--input_crop_data_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_crop/") 
    parser.add_argument('--output_csv_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_crop_csv/") 
    parser.add_argument('--output_error_crop_data_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_error_crop_data/") 
    parser.add_argument('--new_style', action='store_true', default=False) 
    parser.add_argument('--bool_write_error_data', action='store_true', default=False) 
    parser.add_argument('--img_folder', type=str, default="Images") 
    parser.add_argument('--xml_folder', type=str, default="xml") 

    args = parser.parse_args()

    args.input_csv_path = os.path.join(args.input_csv_dir, args.date_name + '.csv')
    args.img_dir = os.path.join(args.input_crop_data_dir, args.date_name, args.img_folder)
    args.xml_dir = os.path.join(args.input_crop_data_dir, args.date_name, args.xml_folder)

    args.output_csv_path = os.path.join(args.output_csv_dir, args.date_name + '.csv')
    args.output_error_crop_data_dir = os.path.join(args.output_error_crop_data_dir, args.date_name)
    args.output_error_data_img_dir = os.path.join(args.output_error_crop_data_dir, args.img_folder)
    args.output_error_data_xml_dir = os.path.join(args.output_error_crop_data_dir, args.xml_folder)

    print("dataset crop csv.")
    print("date_name: {}".format(args.date_name))
    print("input_csv_path: {}".format(args.input_csv_path))
    print("output_csv_path: {}".format(args.output_csv_path))

    # 生成 dataset csv
    dataset_csv(args)

    # 保存 error data
    write_error_data(args)