import cv2
from math import fabs, sin, radians, cos
import numpy as np
import numpy.random as npr
import random


def rotate_img_with_points(image, points, degree):
    """逆时针旋转图像image角度degree，并计算原图中坐标点points在旋转后的图像中的位置坐标.
    Args:
        image: 图像数组
        degree: 旋转角度
        points (np.array): ([x, ...], [y, ...]), shape=(2, m)，原图上的坐标点
    Return:
        new_img: 旋转后的图像
        new_pts: 原图中坐标点points在旋转后的图像中的位置坐标
    """
    h, w = image.shape[:2]
    rotate_center = (w/2, h/2)

    #获取旋转矩阵
    # 参数1为旋转中心点;
    # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
    # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
    M = cv2.getRotationMatrix2D(rotate_center, degree, 1.0)

    #计算图像新边界
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))

    #调整旋转矩阵以考虑平移
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated_img = cv2.warpAffine(image, M, (new_w, new_h))

    a = np.array([M[0][: 2], M[1][: 2]])
    b = np.array([[M[0][2]], [M[1][2]]])

    rotated_points = np.round(np.dot(a, points.astype(np.float32)) + b).astype(np.int64)

    return rotated_img, rotated_points


def aug_rotate_roi(img, img_roi, aug_rotate_angle_list):
    
    # img_roi
    x1 = img_roi[0]
    y1 = img_roi[1]
    x2 = img_roi[2]
    y2 = img_roi[3]
    points = np.array(((x1, x2, x2, x1), (y1, y1, y2, y2)))

    # aug_rotate_angle
    aug_rotate_angle = random.sample(aug_rotate_angle_list, 1)[0]
    
    # rotate_img_with_points
    rotate_img, rotate_points = rotate_img_with_points(img, points, aug_rotate_angle)
    
    # rotate_img_roi
    x1 = min(rotate_points[0][0], rotate_points[0][1], rotate_points[0][2], rotate_points[0][3])
    y1 = min(rotate_points[1][0], rotate_points[1][1], rotate_points[1][2], rotate_points[1][3])
    x2 = max(rotate_points[0][0], rotate_points[0][1], rotate_points[0][2], rotate_points[0][3])
    y2 = max(rotate_points[1][0], rotate_points[1][1], rotate_points[1][2], rotate_points[1][3])
    rotate_img_roi = [x1, y1, x2, y2]

    return rotate_img, rotate_img_roi


def aug_expand_roi(img, img_roi, aug_expand_ratio_list):
    
    # aug_expand_ratio
    aug_expand_ratio = random.sample(aug_expand_ratio_list, 1)[0]

    x1 = img_roi[0]
    y1 = img_roi[1]
    x2 = img_roi[2]
    y2 = img_roi[3]
    h = y2 - y1
    w = x2 - x1

    # expand_img_roi
    # x1 = max(0, x1 + npr.randint(-aug_expand_ratio * w, 1))
    # x2 = min(img.shape[1], x2 + npr.randint(-1, aug_expand_ratio * w))
    # y1 = max(0, y1 + npr.randint(-aug_expand_ratio * h, 1))
    # y2 = min(img.shape[0], y2 + npr.randint(-1, aug_expand_ratio * h))
    x1 = max(0, x1 + npr.randint(-(aug_expand_ratio * w), max(aug_expand_ratio * w, 1)))
    x2 = min(img.shape[1], x2 + npr.randint(min(-aug_expand_ratio * w, -1), aug_expand_ratio * w))
    y1 = max(0, y1 + npr.randint(-aug_expand_ratio * h, max(aug_expand_ratio * h, 1)))
    y2 = min(img.shape[0], y2 + npr.randint(min(-aug_expand_ratio * h, -1), aug_expand_ratio * h))
    expand_img_roi = [x1, y1, x2, y2]

    return expand_img_roi


def aug_rotate_bbox(img, img_roi, object_roi, aug_rotate_angle_list):
    
    # img_roi
    x1 = img_roi[0]
    y1 = img_roi[1]
    x2 = img_roi[2]
    y2 = img_roi[3]
    points = np.array(((x1, x2, x2, x1), (y1, y1, y2, y2)))

    # object_roi
    for idx in range(len(object_roi)):
        bndbox = object_roi[idx]["bndbox"]
        x1 = bndbox[0] + img_roi[0]
        y1 = bndbox[1] + img_roi[1]
        x2 = bndbox[2] + img_roi[0]
        y2 = bndbox[3] + img_roi[1]
        points = np.concatenate((points, np.array(((x1, x2, x2, x1), (y1, y1, y2, y2)))), axis=1)

    # aug_rotate_angle
    aug_rotate_angle = random.sample(aug_rotate_angle_list, 1)[0]
    
    # rotate_img_with_points
    rotate_img, rotate_points = rotate_img_with_points(img, points, aug_rotate_angle)
    
    # rotate_img_roi
    x1 = min(rotate_points[0][0], rotate_points[0][1], rotate_points[0][2], rotate_points[0][3])
    y1 = min(rotate_points[1][0], rotate_points[1][1], rotate_points[1][2], rotate_points[1][3])
    x2 = max(rotate_points[0][0], rotate_points[0][1], rotate_points[0][2], rotate_points[0][3])
    y2 = max(rotate_points[1][0], rotate_points[1][1], rotate_points[1][2], rotate_points[1][3])
    rotate_img_roi = [x1, y1, x2, y2]
    
    # rotate_object_roi
    rotate_object_roi = []
    for idx in range(len(object_roi)):
        classname = object_roi[idx]["classname"]

        x1 = min(rotate_points[0][4 + idx*4], rotate_points[0][4 + idx*4 + 1], rotate_points[0][4 + idx*4 + 2], rotate_points[0][4 + idx*4 + 3]) - rotate_img_roi[0]
        y1 = min(rotate_points[1][4 + idx*4], rotate_points[1][4 + idx*4 + 1], rotate_points[1][4 + idx*4 + 2], rotate_points[1][4 + idx*4 + 3]) - rotate_img_roi[1]
        x2 = max(rotate_points[0][4 + idx*4], rotate_points[0][4 + idx*4 + 1], rotate_points[0][4 + idx*4 + 2], rotate_points[0][4 + idx*4 + 3]) - rotate_img_roi[0]
        y2 = max(rotate_points[1][4 + idx*4], rotate_points[1][4 + idx*4 + 1], rotate_points[1][4 + idx*4 + 2], rotate_points[1][4 + idx*4 + 3]) - rotate_img_roi[1]
        bndbox = [x1, y1, x2, y2]

        rotate_object_roi.append({"classname": classname, "bndbox":bndbox})

    return rotate_img, rotate_img_roi, rotate_object_roi


def aug_expand_bbox(img, img_roi, object_roi, aug_expand_ratio_list):

    # aug_expand_ratio
    aug_expand_ratio = random.sample(aug_expand_ratio_list, 1)[0]

    x1 = img_roi[0]
    y1 = img_roi[1]
    x2 = img_roi[2]
    y2 = img_roi[3]
    h = y2 - y1
    w = x2 - x1

    # expand_img_roi
    x1 = max(0, x1 + npr.randint(-aug_expand_ratio * w, 1))
    x2 = min(img.shape[1], x2 + npr.randint(-1, aug_expand_ratio * w))
    y1 = max(0, y1 + npr.randint(-aug_expand_ratio * h, 1))
    y2 = min(img.shape[0], y2 + npr.randint(-1, aug_expand_ratio * h))
    expand_img_roi = [x1, y1, x2, y2]

    # expand_object_roi
    expand_object_roi = []
    for idx in range(len(object_roi)):
        classname = object_roi[idx]["classname"]
        bndbox = object_roi[idx]["bndbox"]

        x1 = bndbox[0] + img_roi[0] - expand_img_roi[0]
        y1 = bndbox[1] + img_roi[1] - expand_img_roi[1]
        x2 = bndbox[2] + img_roi[0] - expand_img_roi[0]
        y2 = bndbox[3] + img_roi[1] - expand_img_roi[1]
        bndbox = [x1, y1, x2, y2]
        expand_object_roi.append({"classname": classname, "bndbox":bndbox})

    return expand_img_roi, expand_object_roi