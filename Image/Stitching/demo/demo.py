import argparse
from re import T
import cv2
import os
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Stitching.demo.ImageStitching_API import *
from Image.Stitching.utils.folder_tools import *


def image_stitching(args):
    
    img_path_list = get_sub_filepaths_suffix(args.img_dir, suffix='.jpg')
    img_path_list.sort()
    json_path_list = [os.path.join(args.json_dir, str(os.path.basename(img_path)).replace('.jpg', '.json')) for img_path in img_path_list]
    xml_path_list = [os.path.join(args.json_dir, str(os.path.basename(img_path)).replace('.jpg', '.xml')) for img_path in img_path_list]

    # image stitch api
    # load bbox 
    image_stitch_api = ImageSitchApi(args.bkg_path_list, args.bkg_roi_list, img_path_list, json_path_list, 'bbox', 'json')
    # image_stitch_api = ImageSitchApi(args.bkg_path_list, args.bkg_roi_list, img_path_list, json_path_list, 'mask', 'json')
    # image_stitch_api = ImageSitchApi(args.bkg_path_list, args.bkg_roi_list, img_path_list, xml_path_list, 'bbox', 'xml')
    
    # load mask
    image_stitch_api.add_img(img_path_list, json_path_list, 'mask', 'json')

    for idx in range(10):

        # run
        img_stitch, img_stitch_list = image_stitch_api.run()

        output_img_path = os.path.join(args.output_dir, "{}.jpg".format(idx))

        # stich_label_list.append({'label': sitch_pitch_idx['label'], 'bbox': [pitch_roi[0], pitch_roi[1], pitch_roi[2], pitch_roi[3]], 'corner': pitch_corner})
        for img_stitch_idx in range(len(img_stitch_list)):
            bbox = img_stitch_list[img_stitch_idx]['bbox']
            cv2.rectangle(img_stitch, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 0)

            corner = img_stitch_list[img_stitch_idx]['corner']
            corner = corner.reshape((-1, 1, 2)).astype(np.int)
            cv2.polylines(img_stitch, corner, True, (0, 255, 0), 2)
        cv2.imwrite(output_img_path, img_stitch)


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.img_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_json/"
    # args.json_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_json/"
    # args.xml_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_json/"
    # args.img_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/rubbish_img/task1-8mm/"
    # args.json_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/rubbish_img/task1-8mm/"
    # args.xml_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/rubbish_img/task1-8mm/"
    args.img_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/rubbish_img/task2-6mm/"
    args.json_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/rubbish_img/task2-6mm/"
    args.xml_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/rubbish_img/task2-6mm/"

    args.bkg_path_list = ["/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_img/00base.jpg", "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_img/01base.jpg", "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_img/03base.jpg"]
    args.bkg_roi_list = [[[0, 1080], [0, 780], [280, 630], [1310, 700], [1660, 1080], [0, 1080]], [[0, 1080], [0, 780], [280, 630], [1310, 700], [1660, 1080], [0, 1080]], [[0, 1080], [0, 780], [280, 630], [1310, 700], [1660, 1080], [0, 1080]]]
    args.output_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_res/"

    image_stitching(args)


if __name__ == '__main__':
    main()
