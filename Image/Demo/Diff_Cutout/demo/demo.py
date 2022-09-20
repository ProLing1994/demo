import argparse
import cv2
import os
import sys

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Demo.Diff_Cutout.demo.DiffCutout_API import *
from Image.Demo.Diff_Cutout.utils.folder_tools import *


def image_diff_cutout(args):

    # image diff cutout api
    image_diff_cutout_api = DiffCutoutAPI()
    image_diff_cutout_api.load_bkg(args.bkg_dir)

    img_path_list = get_sub_filepaths_suffix(args.img_dir, suffix='.jpg')
    img_path_list.sort()

    for idx in range(len(img_path_list)):
        img_path = img_path_list[idx]
        output_img_path = os.path.join(args.output_dir, "{}.jpg".format(idx))

        img, bbox_list = image_diff_cutout_api.run(img_path)
        
        for idy in range(len(bbox_list)):
            bbox = bbox_list[idy]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.imwrite(output_img_path, img)


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.bkg_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/rubbish_img/jpg/bkg_1/"
    # args.bkg_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/rubbish_img/jpg/bkg_2/"
    args.bkg_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/rubbish_img/jpg/bkg_3/"
    # args.img_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/rubbish_img/jpg/jpg_1/"
    # args.img_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/rubbish_img/jpg/jpg_2/"
    args.img_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/rubbish_img/jpg/jpg_3/"
    args.output_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_res/"

    image_diff_cutout(args)


if __name__ == '__main__':
    main()
