# -*- coding: utf-8 -*-

import cv2
import json
import numpy as np
from copy import deepcopy


### single label (普通单标签多分类，默认无多标签像素)
def json2mask(img, jsonpath, CLASSES):
    # CLASSES: 首位置为背景
    # return mask of shape (h,w,1)
    # nclass don't include background
    # 如有多标签像素，将会按照json对象顺序覆盖
    h, w, c = img.shape
    nclass = len(CLASSES)
    mask = np.zeros((h, w, 3), dtype=img.dtype)
    
    with open(jsonpath, 'r', encoding='utf-8') as fp:
        json_data = json.load(fp )
        for target in json_data['shapes']:
            classname = target['label']
            if classname in CLASSES:
                idx = CLASSES.index(classname)
                value = int(idx)
                contours = np.array(target['points'])
                mask = cv2.drawContours(mask, contours, -1, (value,value,value), cv2.FILLED)
    return mask[..., :1] # h,w,1

def mask2rgb(mask, colormap):
    # mask:(h,w)
    label_colours = np.array(colormap)
    r = mask.copy()
    g = mask.copy()
    b = mask.copy()
    for l in range(0, label_colours.shape[0]):
        r[mask == l] = label_colours[l, 0]
        g[mask == l] = label_colours[l, 1]
        b[mask == l] = label_colours[l, 2]

        rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
    return rgb


### multi label (多标签分割)
def multi_label_mask(img, jsonpath, CLASSES):
    # return mask of shape (h,w,nclass)
    # nclass don't include background
    h, w, c = img.shape
    nclass = len(CLASSES)
    
    mask = np.zeros((h, w, nclass), dtype=img.dtype)
    
    with open(jsonpath, 'r') as fp:
        json_data = json.load(fp)
        for target in json_data['shapes']:
            classname = target['label']
            idx = CLASSES.index(classname)
            contours = [np.array(target['points'])]
            layer = mask[..., idx:idx+1]
            layer = np.repeat(layer, 3, 2)
            layer = cv2.drawContours(layer, contours, -1, (1,1,1), cv2.FILLED)
            mask[..., idx:idx+1] = layer[..., :1]
    return mask

def multi_label_view(img, label, CLASSES, colors):
    # label: multi_label_mask (h,w,nclass)
    # CLASSES: don't include background
    h, w, c = img.shape
    h, w, nclass = label.shape
    assert nclass == len(CLASSES)
    
    rgb_mask_list = multi_mask2rgb(label, nclass, colors)
    for i in range(nclass):
        # mask = label[..., i:i+1]
        # mask = np.repeat(mask, 3, 2)
        mask = rgb_mask_list[i]
        img = cv2.addWeighted(img, 0.8, mask, 0.5, 0)
    return img

def multi_mask2rgb(label, nclass, color_map):
    # mask: (h, w, nclass)
    # nclass don't include background
    # color_map don't include background
    rgb_list = []
    for i in range(nclass):
        mask = label[..., i]
        label_colours = color_map[i]
        
        r = mask.copy()
        g = mask.copy()
        b = mask.copy()
        
        r[mask == 1] = label_colours[0]
        g[mask == 1] = label_colours[1]
        b[mask == 1] = label_colours[2]

        rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        rgb_list.append(rgb)
    return rgb_list


### multi label --> single label (多标签强制单标签分类)
def single_label_mask(img, jsonpath, CLASSES, ORDER):
    # return mask of shape (h,w,1)
    # nclass don't include background
    # 注意：该函数只适用于强行将多标签分类做成多分类
    # 标签图层会根据ORDER顺序覆盖前序层
    h, w, c = img.shape
    mask = np.zeros((h, w, 3), dtype=img.dtype)
    
    with open(jsonpath, 'r') as fp:
        json_data = json.load(fp)
        for order in ORDER:
            for target in json_data['shapes']:
                classname = target['label']
                idx = CLASSES.index(classname)
                if idx == order:
                    value = idx + 1
                    contours = [np.array(target['points'])]
                    mask = cv2.drawContours(mask, contours, -1, (value,value,value), cv2.FILLED)
    return mask[..., :1] # h,w,1

def single_label_view(img, label, colors):
    # label: single_label_mask (h,w,1)
    # colors: [(0,0,0), (0,255,0), ...]
    mask = single_mask2rgb(label, 2, colors)
    img = cv2.addWeighted(img, 0.6, mask, 0.6, 0)
    return img

def single_mask2rgb(mask, nclass, label_colours):
    r = mask.copy()
    g = mask.copy()
    b = mask.copy()
    
    for i in range(nclass+1):
        r[mask == i] = label_colours[i][0]
        g[mask == i] = label_colours[i][1]
        b[mask == i] = label_colours[i][2]

    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[:, :, :1] = r
    rgb[:, :, 1:2] = g
    rgb[:, :, 2:] = b
    return rgb


### others
def rgb2mask(mask, colormap):
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for i, label in enumerate(colormap):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
    label_mask = label_mask.astype(int)
    return label_mask


### mask 2 json
json_info = {
    "shapes":[], 
    "imageHeight": 720,
    "imageWidth": 1280,
    "version": "215.20",
    "flags": {},
    "imagePath": "",
    "imageData": None, 
}
shapes_blank = {
    "label": "",
    "line_color": [0,255,0,128],
    "fill_color": [255,0,0,128],
    "points": [], 
    "z_order": 0,
    "shape_type": "polygon"
}
def contours2json(conts, classes, imgname, jsonpath, imgh, imhw, n_points_filter=10):
    # classes : class needed to write, list
    z_order = 0
    newjson = deepcopy(json_info)
    newjson["imageHeight"] = imgh
    newjson["imageWidth"] = imhw
    newjson['imagePath'] = imgname
    for c in classes:
        boxes = conts[c]
        for i in range(len(boxes)):
            box = boxes[i]
            
            if len(box) >= n_points_filter:
                newbox = deepcopy(shapes_blank)
                newbox['label'] = str(c)
                newbox['z_order'] = z_order
                for j in range(len(box)):
                    x,y = box[j, 0, :]
                    newbox['points'].append([int(x),int(y)])
                
                newjson['shapes'].append(newbox)
                z_order += 1
    
    with open(jsonpath, "w") as f:
        f.write(json.dumps(newjson, ensure_ascii=False, indent=1))
    return newjson



