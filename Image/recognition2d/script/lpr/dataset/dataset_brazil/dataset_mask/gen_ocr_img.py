import argparse
import cv2
import os
import pandas as pd
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def gen_ocr_img(args):

    # mkdir
    create_folder(args.output_img_dir)

    # init 
    csv_list = []           # [{"img_path": "", "json_path": "", "roi_img_path": "", "id": "", "name": "", "roi": "", "color": "", "column": "", "num": ""}]

    # pd
    data_pd = pd.read_csv(args.input_csv_path)

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):

        # info
        img_path = row['img_path']
        json_path = row['json_path']
        plate_id = row['id'] 
        plate_name = row['name']
        plate_roi = row['roi'] 
        plate_color = row['color']
        plate_column = row['column']
        plate_num = row['num']

        plate_img_name = plate_name + '.jpg'
        output_img_path = os.path.join(args.output_img_dir, plate_img_name)

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
            cv2.imwrite(output_img_path, plate_img)
            csv_list.append({"img_path": img_path, "json_path": json_path, "roi_img_path": output_img_path, "id": plate_id, "name": plate_name, "roi": plate_roi, "color": plate_color, "column": plate_column, "num": plate_num})
        except:
            continue

    # out csv
    csv_pd = pd.DataFrame(csv_list)
    csv_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Brazil_new_style") 
    parser.add_argument('--ocr_name', type=str, default="plate_brazil_202308") 
    parser.add_argument('--input_csv_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_csv/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/") 
    args = parser.parse_args()

    args.input_csv_path = os.path.join(args.input_csv_dir, args.date_name + '.csv')
    args.output_dir = os.path.join(args.output_dir, args.ocr_name, args.date_name)

    print("gen ocr img.")
    print("date_name: {}".format(args.date_name))
    print("ocr_name: {}".format(args.ocr_name))
    print("input_csv_path: {}".format(args.input_csv_path))
    print("output_dir: {}".format(args.output_dir))

    args.output_img_dir = os.path.join(args.output_dir, "Images")
    args.output_csv_path = os.path.join(args.output_dir, '{}.csv'.format(args.date_name))

    # 生成 ocr img
    gen_ocr_img(args)
