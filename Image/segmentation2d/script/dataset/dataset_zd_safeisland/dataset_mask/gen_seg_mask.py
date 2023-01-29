import argparse
import cv2
import json
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
# sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *



def get_color(idx):
    idx = idx * 5
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


mask_color_dict = {
    'roadside': get_color(1),
    'rail': get_color(2),
    'roadside_ignore': get_color(3), 
    'rail_ignore': get_color(4), 
    'roadside_rail_nearest': get_color(5), 
}


def drow_ori_mask(annotation, mask, mask_img):
    for track in annotation['shapes']:
        label = track['label']
        points = track['points']
        type = track['type']
        assert type == "polygon"

        contours = []
        for x, y in zip(points[::2], points[1::2]):
            contours.append([x, y])
        contours = [np.array(contours).reshape(-1, 1, 2)]
        
        mask = cv2.drawContours(mask, contours, -1, mask_color_dict[label], cv2.FILLED)
        mask_img = cv2.drawContours(mask_img, contours, -1, mask_color_dict[label], cv2.FILLED)

    return mask, mask_img


def drow_nearest_mask(annotation, mask, mask_img):

    anno_dict = {}
    for track in annotation['shapes']:
        label = track['label']
        points = track['points']
        type = track['type']
        assert type == "polygon"

        contours = []
        for x, y in zip(points[::2], points[1::2]):
            contours.append([x, y])
        contours = [np.array(contours).reshape(-1, 1, 2)]

        if label not in anno_dict:
            anno_dict[label] = []

        anno_dict[label].append(contours)
    
    # 'roadside' 
    if 'roadside' in anno_dict:

        label = 'roadside'
        if len(anno_dict[label]) > 1:
            # 筛选一个最近的标签
            max_center_y = 0
            max_index = 0
            
            for idx in range(len(anno_dict[label])):
                mean_center_y = 0
                
                contour_idx = anno_dict[label][idx][0]
                for idy in range(len(contour_idx)):
                    mean_center_y += contour_idx[idy][0][1]
                
                mean_center_y /= len(contour_idx)
                if mean_center_y > max_center_y:
                    max_center_y = mean_center_y
                    max_index = idx
            
            contours = anno_dict[label][max_index]
        else:
            contours = anno_dict[label][0]

    # 'rail' roadside_ignore' 'rail_ignore'
    else:

        max_center_y = 0
        max_label_index = 'rail'
        max_index = 0

        # 筛选一个最近的标签
        for idz in anno_dict.keys():
            label_idz = idz

            for idx in range(len(anno_dict[label_idz])):
                mean_center_y = 0
                
                contour_idx = anno_dict[label_idz][idx][0]
                for idy in range(len(contour_idx)):
                    mean_center_y += contour_idx[idy][0][1]
                
                mean_center_y /= len(contour_idx)
                if mean_center_y > max_center_y:
                    max_center_y = mean_center_y
                    max_label_index = label_idz
                    max_index = idx
            
            contours = anno_dict[max_label_index][max_index]


    mask = cv2.drawContours(mask, contours, -1, mask_color_dict['roadside_rail_nearest'], cv2.FILLED)
    mask_img = cv2.drawContours(mask_img, contours, -1, mask_color_dict['roadside_rail_nearest'], cv2.FILLED)

    return mask, mask_img


def gen_seg_mask(args):

    # mkdir
    create_folder(args.output_mask_dir)
    create_folder(args.output_mask_img_dir)

    # img_list
    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    for idx in tqdm(range(len(img_list))):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)
        json_path = os.path.join(args.input_json_dir, img_name.replace(".jpg", ".json"))

        output_mask_path = os.path.join(args.output_mask_dir, img_name.replace(".jpg", ".png"))
        output_mask_img_path = os.path.join(args.output_mask_img_dir, img_name.replace(".jpg", ".png"))

        # img
        img = cv2.imread(img_path)
        mask = np.zeros(img.shape, dtype=img.dtype)
        mask_img = np.zeros(img.shape, dtype=img.dtype)

        # read json
        with open(json_path, 'r', encoding='UTF-8') as fr:
            annotation = json.load(fr)

        # draw
        # mask, mask_img = drow_ori_mask(annotation, mask, mask_img)
        mask, mask_img = drow_nearest_mask(annotation, mask, mask_img)

        # mask_img
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.5, gamma=0.)
        
        cv2.imwrite(output_mask_path, mask)
        cv2.imwrite(output_mask_img_path, mask_img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/mnt/huanyuan2/data/image/ZD_SafeIsland/original/base/"
    args.input_img_dir = os.path.join(args.input_dir, "test")
    args.input_json_dir = os.path.join(args.input_dir, "test")

    # args.output_mask_dir = os.path.join(args.input_dir, 'mask_test/mask')
    # args.output_mask_img_dir = os.path.join(args.input_dir, 'mask_test/mask_img')
    args.output_mask_dir = os.path.join(args.input_dir, 'mask_test_nearest/mask')
    args.output_mask_img_dir = os.path.join(args.input_dir, 'mask_test_nearest/mask_img')

    gen_seg_mask(args)