import argparse
import cv2
import os
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from dataset import aug


def test(args):
    
    # mkdir
    create_folder(args.output_img_dir)

    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    for idx in tqdm(range(len(img_list))):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)

        # img
        img = cv2.imread(img_path)

        # aug
        for idy in range(args.aug_times):
            output_img_path = os.path.join(args.output_img_dir, img_name.replace(".jpg", "_aug_{}.png".format(idy)))

            img_aug = aug.imgaug_cityseg(img.copy())
            cv2.imwrite(output_img_path, img_aug)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/kind_num_city_cartype_0804_0809/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/kind_num_city_cartype_0810_0811/"

    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.output_img_dir = os.path.join(args.input_dir, 'img_test')

    args.aug_times = 5

    test(args)