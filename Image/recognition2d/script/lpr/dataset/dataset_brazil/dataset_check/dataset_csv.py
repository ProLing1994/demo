import argparse
import cv2
import io
import json
import os
import pandas as pd
import shutil
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Basic.script.json.platform_json_write import PlatformJsonWriter

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from Image.recognition2d.script.lpr.dataset.dataset_brazil.dataset_dict.dataset_brazil_dict_normal import *


def dataset_csv(args):

    # mkdir
    create_folder(args.output_csv_dir)
    create_folder(args.output_error_data_dir)

    # img list 
    img_list = get_sub_filepaths_suffix(args.img_dir, ".jpg")
    img_list.sort()
    print("data len: ", len(img_list))

    # init 
    csv_list = []           # [{"img_path": "", "json_path": "", "id": "", "name": "", "roi": "", "color": "", "column": "", "num": ""}]
    error_list = []         # [{"img_path": "", "json_path": "", "type": "", "value": ""}]

    for idx in tqdm(range(len(img_list))):

        plate_id = 0
        img_name = os.path.basename(img_list[idx]).replace(".jpg", "")
        img_path = img_list[idx]
        json_name = img_name + ".json"
        json_path = os.path.join(args.json_dir, json_name)

        if args.bool_check_img:
            # img
            try:
                img = cv2.imread(img_path)
            except:
                error_list.append({"img_path": img_path, "json_path": json_path, "type": "img", "value": ""})
                print('"img_path": {}, "json_path": {},"type": "img"'.format(img_path, json_path))
                raise Exception()
                continue

        # json 
        try:
            with io.open(json_path, "r", encoding="UTF-8") as f:
                data_json = json.load(f, encoding='utf-8')
                f.close()
        except:
            error_list.append({"img_path": img_path, "json_path": json_path, "type": "json", "value": ""})
            print('"img_path": {}, "json_path": {},"type": "img"'.format(img_path, json_path))
            raise Exception()
            continue
        
        for cell in data_json['shapes']:

            # plate num
            plate_num = json_load_object_plate_num(cell)
            if 'none' in plate_num.lower() or plate_num.lower() == '':
                continue

            # plate roi
            try:
                x1, x2, y1, y2, w, h = json_load_object_plate_points(cell)
                plate_roi = str("{},{},{},{}".format(x1, x2, y1, y2))
                if args.bool_check_img:
                    plate_img = img[y1:y2, x1:x2]
            except:
                error_list.append({"img_path": img_path, "json_path": json_path, "type": "plate_roi", "value": ""})
                print('"img_path": {}, "json_path": {},"type": "plate_roi"'.format(img_path, json_path))
                raise Exception()
                continue
            
            # plate status
            plate_status = json_load_object_plate_status(cell)
            if plate_status != status_name_list[0]:
                continue

            # plate color
            plate_color, load_plate_color = json_load_object_plate_color(cell)
            # if plate_color == color_name_list[0]:
            #     error_list.append({"img_path": img_path, "json_path": json_path, "type": "plate_color", "value": load_plate_color})
            #     # print('"img_path": {}, "json_path": {},"type": "plate_color", "value": {}'.format(img_path, json_path, load_plate_color))
            #     # raise Exception()
            #     continue

            # plate column
            plate_column, load_plate_column = json_load_object_plate_column(cell, w, h)
            if plate_column == column_name_list[0]:
                error_list.append({"img_path": img_path, "json_path": json_path, "type": "plate_column", "value": load_plate_column})
                print('"img_path": {}, "json_path": {},"type": "plate_column", "value": {}'.format(img_path, json_path, load_plate_column))
                raise Exception()
                continue

            # plate num check
            # 普通车牌：7
            if not ((len(plate_num) == 7)):
                error_list.append({"img_path": img_path, "json_path": json_path, "type": "plate_num", "value": plate_num})
                print('"img_path": {}, "json_path": {},"type": "plate_num", "value": {}'.format(img_path, json_path, plate_num))
                continue
            
            # 普通车牌：前三位必须为字母
            bool_error_plate_num = np.array([True if str_num not in kind_num_labels[11:] else False for str_num in plate_num[:3]]).sum()
            if bool_error_plate_num:
                error_list.append({"img_path": img_path, "json_path": json_path, "type": "plate_num", "value": plate_num})
                print('"img_path": {}, "json_path": {},"type": "plate_num", "value": {}'.format(img_path, json_path, plate_num))
                continue  
            
            # 普通车牌：第四、六、七位必须为数字
            bool_error_plate_num = np.array([True if str_num not in kind_num_labels[:11] else False for str_num in plate_num[3]]).sum()
            bool_error_plate_num += np.array([True if str_num not in kind_num_labels[:11] else False for str_num in plate_num[5]]).sum()
            bool_error_plate_num += np.array([True if str_num not in kind_num_labels[:11] else False for str_num in plate_num[6]]).sum()
            if bool_error_plate_num:
                error_list.append({"img_path": img_path, "json_path": json_path, "type": "plate_num", "value": plate_num})
                print('"img_path": {}, "json_path": {},"type": "plate_num", "value": {}'.format(img_path, json_path, plate_num))
                continue  
            
            # 普通车牌：第五位可以为数字和字母
            # # 当前阶段，如果是 1、0、I、O 进行检修
            # bool_error_plate_num = np.array([True if str_num in ['1', '0', 'I', 'O'] else False for str_num in plate_num[4]]).sum()
            # if bool_error_plate_num:
            #     error_list.append({"img_path": img_path, "json_path": json_path, "type": "plate_num", "value": plate_num})
            #     print('"img_path": {}, "json_path": {},"type": "plate_num", "value": {}'.format(img_path, json_path, plate_num))
            #     continue  

            plate_name = args.data_format.format(img_name, plate_id, plate_color, plate_column, plate_num)
            plate_id += 1

            csv_list.append({"img_path": img_path, "json_path": json_path, "id": plate_id, "name": plate_name, "roi": plate_roi, "color": plate_color, "column": plate_column, "num": plate_num})

    # out csv
    csv_pd = pd.DataFrame(csv_list)
    csv_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")

    error_data_csv_path = os.path.join(args.output_error_data_dir, 'error.csv')
    error_pd = pd.DataFrame(error_list)
    error_pd.to_csv(error_data_csv_path, index=False, encoding="utf_8_sig")


def write_crop_data(args):

    if not args.bool_write_crop_data:
        return

    # mkdir 
    create_folder(args.output_crop_data_img_dir)

    # pd
    data_pd = pd.read_csv(args.output_csv_path)

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
        
        # info
        img_path = row['img_path']
        plate_name = row['name']
        plate_roi = row['roi'] 

        # img
        img = cv2.imread(img_path)
        
        # plate_img
        plate_roi_list = plate_roi.split(',')
        x1 = int(plate_roi_list[0])
        x2 = int(plate_roi_list[1])
        y1 = int(plate_roi_list[2])
        y2 = int(plate_roi_list[3])
        plate_img = img[y1:y2, x1:x2]

        try:
            cv2.imwrite(os.path.join(args.output_crop_data_img_dir, plate_name + ".jpg"), plate_img)
        except:
            continue


def write_error_data(args):

    if not args.bool_write_error_data:
        return
    
    # init
    platform_json_writer = PlatformJsonWriter()

    # mkdir
    create_folder(args.output_error_data_img_dir)
    create_folder(args.output_error_data_json_dir)
    
    # pd
    error_data_csv_path = os.path.join(args.output_error_data_dir, 'error.csv')

    try:
        data_pd = pd.read_csv(error_data_csv_path)
    except:
        return

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        # init 
        rect_list = []

        # info
        img_path = row['img_path']
        json_path = row['json_path']
        img_name = os.path.basename(img_path)
        json_name = os.path.basename(json_path)

        # img
        try:
            to_img_path = os.path.join(args.output_error_data_img_dir, img_name)
            shutil.copy(img_path, to_img_path)
        except:
            continue
        
        # json
        try:
            with io.open(json_path, "r", encoding="UTF-8") as f:
                data_json = json.load(f, encoding='utf-8')
                f.close()
        except:
            continue


        for cell in data_json['shapes']:

            try:
                # plate num
                plate_num = json_load_object_plate_num(cell)
                if 'none' in plate_num.lower() or plate_num.lower() == '':
                    continue

                # plate points
                x1, x2, y1, y2, w, h = json_load_object_plate_points(cell)

                # plate color
                plate_color, _ = json_load_object_plate_color(cell)
                if plate_color == color_name_list[0]:
                    plate_color = "unknown"

                # plate column
                plate_column, _ = json_load_object_plate_column(cell, w, h)

            except:
                continue
            
            rect_list.append([x1, y1, x2, y2, "license", [{"name": "number", "value": plate_num}, {"name": "color", "value": plate_color}, {"name": "column", "value": plate_column}, {"name": "type", "value": "car"}, {"name": "status", "value": "n"}]])

        try:
            out_json_path = os.path.join(args.output_error_data_json_dir, json_name)
            # platform_json_writer.write_json(img.shape[1], img.shape[0], img_name, out_json_path, frame_num=idx, rect_list=rect_list)
            platform_json_writer.write_json(2592, 1920, img_name, out_json_path, frame_num=idx, rect_list=rect_list)
        except:
            continue



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="brazil_02210_202301") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil/") 
    parser.add_argument('--output_csv_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_csv/") 
    parser.add_argument('--output_crop_data_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_crop/") 
    parser.add_argument('--output_error_data_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_error_data/") 
    parser.add_argument('--bool_write_crop_data', action='store_true', default=False) 
    parser.add_argument('--bool_write_error_data', action='store_true', default=False) 
    parser.add_argument('--bool_check_img', action='store_true', default=False) 
    parser.add_argument('--img_folder', type=str, default="JPEGImages") 
    parser.add_argument('--json_folder', type=str, default="Json") 
    args = parser.parse_args()

    args.data_format = "{}-{:0>2d}_{}_{}_{}"        # name-id_颜色_单双行_车牌号
    args.img_dir = os.path.join(args.input_dir, args.date_name, args.img_folder)
    args.json_dir = os.path.join(args.input_dir, args.date_name, args.json_folder)
    args.output_csv_path = os.path.join(args.output_csv_dir, args.date_name + '.csv')
    args.output_crop_data_dir = os.path.join(args.output_crop_data_dir, args.date_name)
    args.output_error_data_dir = os.path.join(args.output_error_data_dir, args.date_name)

    print("dataset csv.")
    print("date_name: {}".format(args.date_name))
    print("input_dir: {}".format(args.img_dir))
    print("output_csv_path: {}".format(args.output_csv_path))

    # 生成 dataset csv
    dataset_csv(args)

    # 保存 crop data
    args.output_crop_data_img_dir = os.path.join(args.output_crop_data_dir, "Images")
    write_crop_data(args)

    # 保存 error data
    args.output_error_data_img_dir = os.path.join(args.output_error_data_dir, "Images")
    args.output_error_data_json_dir = os.path.join(args.output_error_data_dir, "Json")
    write_error_data(args)