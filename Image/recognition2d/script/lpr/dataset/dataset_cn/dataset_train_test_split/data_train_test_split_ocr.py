import argparse
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

from sklearn.model_selection import train_test_split

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def split(args):
    
    # mkdir
    create_folder(os.path.dirname(args.trainval_file))

    # init
    csv_list = []           # [{"img_path": "", "json_path": "", "roi_img_path": "", "id": "", "name": "", "roi": "", "color": "", "column": "", "num": "", "split_type": ""}]
    file_list = []          # ["num"]

    # pd
    data_pd = pd.read_csv(args.input_csv_path)

    # file_list
    for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
        plate_num  = row['num']
        if plate_num not in file_list:
            file_list.append(plate_num)

    # train_test_split
    trainval_list, test_list = train_test_split(file_list, test_size=args.test_size, random_state=0)
    train_list, val_list = train_test_split(trainval_list, test_size=args.val_size, random_state=0)
    print("length: trainval: {}, train: {}, val: {}, test: {}, all: {}".format(len(trainval_list), len(train_list), len(val_list), len(test_list), (len(train_list) + len(val_list) + len(test_list))))

    # csv_list
    for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
        plate_num  = row['num']
        split_type = ""
        if plate_num in train_list:
            split_type = "train"
        elif plate_num in val_list:
            split_type = "val"
        elif plate_num in test_list:
            split_type = "test"
        else:
            raise Exception
        row['split_type'] = split_type
        csv_list.append(row)

    with open(args.trainval_file, "w") as f:

        for idx in range(len(csv_list)):
            file = csv_list[idx]
            if file['split_type'] == "train" or file['split_type'] == "val":
                f.write('{}'.format(file["roi_img_path"]))
                f.write("\n")

    with open(args.train_file, "w") as f:
        for idx in range(len(csv_list)):
            file = csv_list[idx]
            if file['split_type'] == "train":
                f.write('{}'.format(file["roi_img_path"]))
                f.write("\n")

    with open(args.val_file, "w") as f:
        for idx in range(len(csv_list)):
            file = csv_list[idx]
            if file['split_type'] == "val":
                f.write('{}'.format(file["roi_img_path"]))
                f.write("\n")

    with open(args.test_file, "w") as f:
        for idx in range(len(csv_list)):
            file = csv_list[idx]
            if file['split_type'] == "test":
                f.write('{}'.format(file["roi_img_path"]))
                f.write("\n")

    # out csv
    csv_pd = pd.DataFrame(csv_list)
    csv_pd.to_csv(args.input_csv_path, columns=["img_path", "json_path", "roi_img_path", "id", "name", "roi", "color", "column", "num", "split_type"], index=False, encoding="utf_8_sig")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="shanxi_jin") 
    parser.add_argument('--ocr_name', type=str, default="plate_cn_202305") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/")  
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.ocr_name, args.date_name)

    print("data train test split.")
    print("date_name: {}".format(args.date_name))
    print("ocr_name: {}".format(args.ocr_name))
    print("input_dir: {}".format(args.input_dir))

    args.input_csv_path = os.path.join(args.input_dir, '{}.csv'.format(args.date_name))
    args.trainval_file = os.path.join(args.input_dir, "ImageSets/Main/trainval.txt")
    args.train_file = os.path.join(args.input_dir, "ImageSets/Main/train.txt")
    args.val_file = os.path.join(args.input_dir, "ImageSets/Main/val.txt")
    args.test_file = os.path.join(args.input_dir, "ImageSets/Main/test.txt")
    
    args.test_size = 0.1
    args.val_size = 0.1

    split(args)