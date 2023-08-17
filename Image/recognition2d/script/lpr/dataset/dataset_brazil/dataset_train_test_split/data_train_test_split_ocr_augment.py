import argparse
import os
import pandas as pd
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def split(args):

    # mkdir
    create_folder(os.path.dirname(args.to_trainval_file))

    # pd
    data_pd = pd.read_csv(args.input_csv_path)

    with open(args.to_trainval_file, "w") as f:

        for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
            if row['split_type'] == "train" or row['split_type'] == "val":
                f.write('{}'.format(row["roi_img_path"]))
                f.write("\n")

    with open(args.to_train_file, "w") as f:

        for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
            if row['split_type'] == "train":
                f.write('{}'.format(row["roi_img_path"]))
                f.write("\n")

    with open(args.to_val_file, "w") as f:

        for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
            if row['split_type'] == "val":
                f.write('{}'.format(row["roi_img_path"]))
                f.write("\n")

    with open(args.to_test_file, "w") as f:

        for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
            if row['split_type'] == "test":
                f.write('{}'.format(row["roi_img_path"]))
                f.write("\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Brazil_new_style") 
    parser.add_argument('--ocr_name', type=str, default="plate_brazil_202308") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/")  
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.ocr_name, args.date_name)

    print("data train test split augment.")
    print("date_name: {}".format(args.date_name))
    print("ocr_name: {}".format(args.ocr_name))
    print("input_dir: {}".format(args.input_dir))

    args.input_csv_path = os.path.join(args.input_dir, args.date_name + '_augument.csv')
    args.to_trainval_file = os.path.join(args.input_dir, "ImageSets_aug/ImageSets/Main/trainval.txt")
    args.to_train_file = os.path.join(args.input_dir, "ImageSets_aug/ImageSets/Main/train.txt")
    args.to_val_file = os.path.join(args.input_dir, "ImageSets_aug/ImageSets/Main/val.txt")
    args.to_test_file = os.path.join(args.input_dir, "ImageSets_aug/ImageSets/Main/test.txt")
    
    args.test_size = 0.1
    args.val_size = 0.1

    split(args)