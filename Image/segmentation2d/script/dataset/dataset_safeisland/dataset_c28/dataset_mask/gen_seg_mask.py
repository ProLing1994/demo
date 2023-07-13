import argparse
import cv2
import json
import numpy as np
import os
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def get_color(idx):
    idx = idx * 5
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


mask_label_dict = {
    'road_boundary': (1, 1, 1),
}

mask_color_dict = {
    'road_boundary': get_color(1),
}


def drow_mask(annotation, mask, mask_img):
    for track in annotation['shapes']:
        label = track['label']
        points = track['points']
        type = track['type']
        if type != "polygon":
            continue

        # contours = []
        # for x, y in zip(points[::2], points[1::2]):
        #     contours.append([x, y])
        # contours = [np.array(contours).reshape(-1, 1, 2)]
        
        # mask = cv2.drawContours(mask, contours, -1, mask_color_dict[label], cv2.FILLED)
        # mask_img = cv2.drawContours(mask_img, contours, -1, mask_color_dict[label], cv2.FILLED)

        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], False, mask_label_dict[label], 11)
        cv2.polylines(mask_img, [pts], False, mask_color_dict[label], 11)

    return mask, mask_img


def gen_seg_mask(args):

    # mkdir
    create_folder(args.mask_dir)
    create_folder(args.mask_img_dir)

    # img_list
    img_list = get_sub_filepaths_suffix(args.img_dir, args.img_suffix)
    img_list.sort()

    for idx in tqdm(range(len(img_list))):

        img_path = img_list[idx]
        img_name = os.path.basename(img_path)

        json_name = img_name.replace(args.img_suffix, args.json_suffix)
        json_path = os.path.join(args.json_dir, json_name)

        mask_name = img_name.replace(args.img_suffix, args.mask_suffix)
        output_mask_path = os.path.join(args.mask_dir, mask_name)
        output_mask_img_path = os.path.join(args.mask_img_dir, mask_name)

        # img
        img = cv2.imread(img_path)

        # mask
        mask = np.zeros(img.shape, dtype=img.dtype)
        mask_img = np.zeros(img.shape, dtype=img.dtype)

        # read json
        with open(json_path, 'r', encoding='UTF-8') as fr:
            annotation = json.load(fr)

        # draw
        mask, mask_img = drow_mask(annotation, mask, mask_img)

        # mask_img
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.5, gamma=0.)
        
        cv2.imwrite(output_mask_path, mask)
        cv2.imwrite(output_mask_img_path, mask_img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="america") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_C28_safeisland/original") 
    parser.add_argument('--img_folder', type=str, default="JPEGImages") 
    parser.add_argument('--json_folder', type=str, default="Jsons") 
    parser.add_argument('--mask_folder', type=str, default="mask") 
    parser.add_argument('--mask_img_folder', type=str, default="mask_img") 
    args = parser.parse_args()

    args.img_suffix = ".jpg"
    args.json_suffix = ".json"
    args.mask_suffix = ".png"
    args.img_dir = os.path.join(args.input_dir, args.date_name, args.img_folder)
    args.json_dir = os.path.join(args.input_dir, args.date_name, args.json_folder)
    args.mask_dir = os.path.join(args.input_dir, args.date_name, args.mask_folder)
    args.mask_img_dir = os.path.join(args.input_dir, args.date_name, args.mask_img_folder)

    print("gen seg mask.")
    print("date_name: {}".format(args.date_name))
    print("img_dir: {}".format(args.img_dir))
    print("json_dir: {}".format(args.json_dir))
    print("mask_dir: {}".format(args.mask_dir))
    print("mask_img_dir: {}".format(args.mask_img_dir))

    gen_seg_mask(args)