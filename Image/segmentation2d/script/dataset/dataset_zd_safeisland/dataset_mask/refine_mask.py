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
from Image.Basic.script.json.json_write import write_json


mask_color_dict = {
    'road_boundary': (0, 0, 255), 
}

def drow_nearest_mask(annotation, mask, mask_img):

    # [{key: , id: , points: }]
    anno_list = []
    for track in annotation['shapes']:
        anno_dict = {}

        label = track['label']
        points = track['points']
        type = track['type']
        assert type == "polygon"

        contours = []
        for x, y in zip(points[::2], points[1::2]):
            contours.append([x, y])
        contours = [np.array(contours).reshape(-1, 1, 2)]
        
        contour_mask = np.zeros(mask.shape, dtype=mask.dtype)
        contour_mask = cv2.drawContours(contour_mask, contours, -1, (255, 255, 255), 1)
        
        y, x = np.where(contour_mask[:, :, 0] == 255)
        anno_dict['key'] = label
        anno_dict['id'] = len(anno_list)
        anno_dict['x'] = x
        anno_dict['y'] = y
        anno_list.append(anno_dict)
    
    res_anno_list = []
    for x_idx in range(mask.shape[1]):

        # 寻找 x 上最大的 y 坐标
        max_id = -1
        max_y = -1
        for idy in range(len(anno_list)):
            anno_idy = anno_list[idy]

            if x_idx < np.array(anno_idy['x']).min() or x_idx > np.array(anno_idy['x']).max():
                continue

            for idz in range(len(anno_idy['x'])):
                if anno_idy['x'][idz] == x_idx:
                    if anno_idy['y'][idz] > max_y:
                        max_y = anno_idy['y'][idz]
                        max_id = anno_idy['id']
        
        # 生成结果
        if max_id != -1:
            anno_id = -1
            bool_find_id = False
            for idz in range(len(res_anno_list)):
                if res_anno_list[idz]['id'] == max_id:
                    bool_find_id = True
                    anno_id = idz
                    break
            
            if bool_find_id == False:
                anno_dict = {}
                anno_dict['key'] = anno_list[max_id]['key']
                anno_dict['id'] = max_id
                anno_dict['x'] = [x_idx]
                anno_dict['y'] = [max_y]
                res_anno_list.append(anno_dict)
            else:
                res_anno_list[anno_id]['x'].append(x_idx)
                res_anno_list[anno_id]['y'].append(max_y)

    # 绘制图像
    for idx in range(len(res_anno_list)):
        contours = []
        for x, y in zip(res_anno_list[idx]['x'], res_anno_list[idx]['y']):
            contours.append([x, y])
        contours = [np.array(contours).reshape(-1, 1, 2)]
        
        # mask = cv2.drawContours(mask, contours, -1, mask_color_dict['road_boundary'], 3)
        # mask_img = cv2.drawContours(mask_img, contours, -1, mask_color_dict['road_boundary'], 3)

        for coor in contours[0]:
            # print(coor)
            mask = cv2.circle(mask, (int(coor[0][0]),int(coor[0][1])), 2, mask_color_dict['road_boundary'], 3)
            mask_img = cv2.circle(mask_img, (int(coor[0][0]),int(coor[0][1])), 2, mask_color_dict['road_boundary'], 3)

    # 保存 json 文件
    json_bboxes = {}
    label = "road_boundary"

    for idx in range(len(res_anno_list)):
        points_list = []
        for x, y in zip(res_anno_list[idx]['x'], res_anno_list[idx]['y']):
            points_list.append([int(x), int(y)])

        if label not in json_bboxes:
            json_bboxes[label] = []     
        json_bboxes[label].append(points_list)

    return mask, mask_img, json_bboxes


def road_boundary(args):

    # mkdir
    create_folder(args.output_mask_dir)
    create_folder(args.output_mask_img_dir)
    create_folder(args.output_json_dir)

    # img_lis
    img_list = get_sub_filepaths_suffix(args.input_img_dir, ".jpg")
    img_list.sort()

    for idx in tqdm(range(len(img_list))):
        img_path = img_list[idx]
        img_name = os.path.basename(img_path)
        json_name = img_name.replace(".jpg", ".json")
        json_path = os.path.join(args.input_json_dir, json_name)

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
        mask, mask_img, json_bboxes = drow_nearest_mask(annotation, mask, mask_img)

        # mask_img
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.5, gamma=0.)
        
        # cv2.imwrite(output_mask_path, mask)
        # cv2.imwrite(output_mask_img_path, mask_img)
        
        # json
        output_json_path = os.path.join(args.output_json_dir, json_name)
        write_json(output_json_path, img_name, img.shape, json_bboxes, "polygon")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/mnt/huanyuan2/data/image/ZD_SafeIsland/original/base/"
    args.input_img_dir = os.path.join(args.input_dir, "train")
    args.input_json_dir = os.path.join(args.input_dir, "Platform_Json_ori")

    args.output_mask_dir = os.path.join(args.input_dir, 'mask_train_nearest/mask')
    args.output_mask_img_dir = os.path.join(args.input_dir, 'mask_train_nearest/mask_img')
    args.output_json_dir = os.path.join(args.input_dir, 'Json_road_boundary')

    road_boundary(args)