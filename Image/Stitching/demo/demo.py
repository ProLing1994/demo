import argparse
import cv2
import os
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Stitching.demo.ImageStitching_API import *


def image_stitching(args):
    
    # image stitch api
    image_stitch_api = ImageSitchApi(args.bkg_path, args.bkg_roi)

    for idx in range(10):

        # img
        img = cv2.imread(args.img_path)

        # run
        # img, img_list = image_stitch_api.run(img, args.json_path, 'json')
        img_stitch, img_list = image_stitch_api.run(img, args.xml_path, 'xml')

        output_img_path = os.path.join(args.output_dir, "{}.jpg".format(idx))
        cv2.imwrite(output_img_path, img_stitch)


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.img_path = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_json/frame_0000002500.jpg"
    args.json_path = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_json/frame_0000002500.json"
    args.xml_path = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_json/frame_0000002500.xml"

    args.bkg_path = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_img/00base.jpg"
    args.bkg_roi = [[0, 1080], [0, 780], [280, 630], [1310, 700], [1660, 1080], [0, 1080]]
    args.output_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_res/"

    image_stitching(args)


if __name__ == '__main__':
    main()
