import argparse
import os
import pandas as pd
import sys
import shutil
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def gen_ocr_img(args):

    # mkdir
    create_folder(args.output_img_dir)

    # init 
    csv_list = []           # [{"img_path": "", "json_path": "", "roi_img_path": "", "id": "", "name": "", "roi": "", "color": "", "column": "", "num": ""}]
    
    # img list
    img_list = get_sub_filepaths_suffix(args.input_dir, 'jpg')
    img_list.sort()

    for idx in tqdm(range(len(img_list))):

        img_path = img_list[idx]
        img_name = os.path.basename(img_path)

        # info
        img_path = img_path
        json_path = ''
        plate_id = 0
        plate_name = img_name.split('.jpg')[0]
        plate_roi = []
        plate_color = ''
        # plate_column = ''
        plate_column = 'single'
        plate_num = str(img_name).split('.')[0].split('_')[-1]
        print(plate_num)

        # plate_img_name = str(plate_name) + '.jpg'
        plate_img_name = str(plate_name).replace(plate_num, "{}_{}".format(plate_column, plate_num)) + '.jpg'
        output_img_path = os.path.join(args.output_img_dir, plate_img_name)

        shutil.copy(img_path, output_img_path)
        csv_list.append({"img_path": img_path, "json_path": json_path, "roi_img_path": output_img_path, "id": plate_id, "name": plate_name, "roi": plate_roi, "color": plate_color, "column": plate_column, "num": plate_num})

    # out csv
    csv_pd = pd.DataFrame(csv_list)
    csv_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="original_563_new_style") 
    parser.add_argument('--ocr_name', type=str, default="plate_brazil_202309") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/DIFFSTE/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/") 
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.date_name)
    args.output_dir = os.path.join(args.output_dir, args.ocr_name, args.date_name)

    print("gen ocr img.")
    print("date_name: {}".format(args.date_name))
    print("ocr_name: {}".format(args.ocr_name))
    print("input_dir: {}".format(args.input_dir))
    print("output_dir: {}".format(args.output_dir))

    args.output_img_dir = os.path.join(args.output_dir, "Images")
    args.output_csv_path = os.path.join(args.output_dir, '{}.csv'.format(args.date_name))

    # 生成 ocr img
    gen_ocr_img(args)
