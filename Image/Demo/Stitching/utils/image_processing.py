import cv2
from math import fabs, sin, radians, cos
import numpy as np
import random


def otsu(img, erode_iter=1, dilate_iter=6):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_OTSU)

    mask = cv2.erode(mask, None, iterations=erode_iter)
    mask = cv2.dilate(mask, None, iterations=dilate_iter)

    mask = np.expand_dims(mask, axis=2)

    return img * mask, mask


def gaussian_otsu(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    retval, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_OTSU)

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=6)

    mask = np.expand_dims(mask, axis=2)

    return img * mask, mask


def gen_gaussian_pyramid(img):

    gp_img = img.copy()
    gp_img_list = [gp_img]
    for i in range(6):
        gp_img = cv2.pyrDown(gp_img)
        gp_img_list.append(gp_img)
    
    return gp_img_list
    

def gen_laplacian_pyramid(gp_img_list):

    lp_img_list = [ gp_img_list[-1] ]
    for i in range(len(gp_img_list) - 1, 0, -1):
        gp_img = cv2.pyrUp(gp_img_list[i])
        lp_img = cv2.subtract(gp_img_list[i-1], gp_img)
        lp_img_list.append(lp_img)
    return lp_img_list


def sitch_laplace_pyramid(bkg_img, sitch_img, sitch_mask):
    """
    https://blog.csdn.net/weixin_41122036/article/details/103071247
    """
    bkg_img = cv2.resize(bkg_img, (1920, 1088))
    sitch_img = cv2.resize(sitch_img, (1920, 1088))
    sitch_mask = cv2.resize(sitch_mask, (1920, 1088))

    gp_bkg_img_list = gen_gaussian_pyramid(bkg_img)
    gp_sitch_img_list = gen_gaussian_pyramid(sitch_img)
    gp_sitch_mask_list = gen_gaussian_pyramid(sitch_mask)
    lp_bkg_img_list = gen_laplacian_pyramid(gp_bkg_img_list)
    lp_sitch_img_list = gen_laplacian_pyramid(gp_sitch_img_list)

    lp_sitch_res_list = []
    for idx in range(len(lp_bkg_img_list)):
        lp_bkg_img_idx, lp_sitch_img_idx = lp_bkg_img_list[idx] , lp_sitch_img_list[idx]
        lp_bkg_img_idx = lp_bkg_img_idx * ( 1 - gp_sitch_mask_list[len(gp_sitch_mask_list) - idx - 1] )
        lp_sitch_res = cv2.addWeighted(src1=lp_bkg_img_idx, alpha=1.0, src2=lp_sitch_img_idx, beta=1.0, gamma=0.)
        lp_sitch_res_list.append(lp_sitch_res)

    sitch_res_img = lp_sitch_res_list[0]
    for i in range(1, len(lp_sitch_img_list)):
        sitch_res_img = cv2.pyrUp(sitch_res_img)
        sitch_res_img = cv2.add(sitch_res_img, lp_sitch_res_list[i])

    sitch_res_img = cv2.resize(sitch_res_img, (1920, 1080))

    return sitch_res_img


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


def sitch_pitch_aug_rotate_bbox(image, bbox, sitch_pitch_aug_rotate_angle_list):

    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    points = np.array(((x1, x2, x2, x1), (y1, y1, y2, y2)))

    sitch_pitch_aug_rotate_angle = random.sample(sitch_pitch_aug_rotate_angle_list, 1)[0]
    rotate_img, rotate_points = rotate_img_with_points(image, points, sitch_pitch_aug_rotate_angle)

    x1 = min(rotate_points[0][0], rotate_points[0][1], rotate_points[0][2], rotate_points[0][3])
    y1 = min(rotate_points[1][0], rotate_points[1][1], rotate_points[1][2], rotate_points[1][3])
    x2 = max(rotate_points[0][0], rotate_points[0][1], rotate_points[0][2], rotate_points[0][3])
    y2 = max(rotate_points[1][0], rotate_points[1][1], rotate_points[1][2], rotate_points[1][3])

    rotate_bbox = [x1, y1, x2, y2]
    return rotate_img, rotate_bbox


def sitch_pitch_aug_mirror_bbox(image, boxes, sitch_pitch_aug_rotate_mirror_list):

    # init
    mirror_bbox = np.array(boxes).copy()
    mirror_img = image

    _, width, _ = image.shape
    if random.sample(sitch_pitch_aug_rotate_mirror_list, 1)[0]:
        mirror_img = image[:, ::-1]
        mirror_bbox[0::2] = width - mirror_bbox[2::-2]

    # contiguous
    mirror_img = np.ascontiguousarray(mirror_img)
    return mirror_img, list(mirror_bbox)
    

def sitch_pitch_aug_scale_bbox(image, boxes, sitch_pitch_aug_rotate_scale_list):

    # init
    scale_bbox = np.array(boxes).copy()
    scale_image = image

    sitch_pitch_aug_rotate_scale = random.sample(sitch_pitch_aug_rotate_scale_list, 1)[0]

    h, w = image.shape[:2]
    h_scale, w_scale = int(h * sitch_pitch_aug_rotate_scale), int(w * sitch_pitch_aug_rotate_scale)
    scale_image = cv2.resize(image, (w_scale, h_scale))
    scale_bbox[0::2] = scale_bbox[0::2] / w * w_scale
    scale_bbox[1::2] = scale_bbox[1::2] / h * h_scale
    scale_bbox = scale_bbox.astype(np.int32)

    return scale_image, list(scale_bbox)


def sitch_pitch_aug_rotate_mask(image, corner, sitch_pitch_aug_rotate_angle_list):
    
    corner = np.array(corner)
    points = np.array((corner[:, 0 ], corner[:, 1]))

    sitch_pitch_aug_rotate_angle = random.sample(sitch_pitch_aug_rotate_angle_list, 1)[0]
    rotate_img, rotate_points = rotate_img_with_points(image, points, sitch_pitch_aug_rotate_angle)

    rotate_corner = np.transpose(rotate_points)

    return rotate_img, rotate_corner


def sitch_pitch_aug_mirror_mask(image, corner, sitch_pitch_aug_rotate_mirror_list):
    
    # init
    mirror_corner = np.array(corner).copy()
    mirror_img = image

    _, width, _ = image.shape
    if random.sample(sitch_pitch_aug_rotate_mirror_list, 1)[0]:
        mirror_img = image[:, ::-1]
        mirror_corner[:, 0] = width - mirror_corner[:, 0]

    # contiguous
    mirror_img = np.ascontiguousarray(mirror_img)
    return mirror_img, mirror_corner


def sitch_pitch_aug_scale_mask(image, corner, sitch_pitch_aug_rotate_scale_list):
    
    # init
    scale_corner = np.array(corner).copy()
    scale_image = image

    sitch_pitch_aug_rotate_scale = random.sample(sitch_pitch_aug_rotate_scale_list, 1)[0]

    h, w = image.shape[:2]
    h_scale, w_scale = int(h * sitch_pitch_aug_rotate_scale), int(w * sitch_pitch_aug_rotate_scale)
    scale_image = cv2.resize(image, (w_scale, h_scale))
    scale_corner[:, 0] = scale_corner[:, 0] / w * w_scale
    scale_corner[:, 1] = scale_corner[:, 1] / h * h_scale
    scale_corner = scale_corner.astype(np.int32)

    return scale_image, scale_corner


def sitch_pitch_foreground_extract(img, sitch_pitch_processing_method_list):
    
    sitch_pitch_processing_method = random.sample(sitch_pitch_processing_method_list, 1)[0]
    
    if sitch_pitch_processing_method == "":
        img = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.ones(gray.shape)
        mask = np.expand_dims(mask, axis=2)
        mask = mask.astype(np.uint8)
    elif sitch_pitch_processing_method == "otsu":
        img, mask = otsu(img)

        if mask.sum() == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = np.ones(gray.shape)
            mask = np.expand_dims(mask, axis=2)
            mask = mask.astype(np.uint8)
    elif sitch_pitch_processing_method == "Gaussian&otsu":
        img, mask = gaussian_otsu(img)

        if mask.sum() == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = np.ones(gray.shape)
            mask = np.expand_dims(mask, axis=2)
            mask = mask.astype(np.uint8)
            
    # 获得更加紧致的边界框
    y1 = min(np.where(mask != 0)[0])
    y2 = max(np.where(mask != 0)[0])
    x1 = min(np.where(mask != 0)[1])
    x2 = max(np.where(mask != 0)[1])    

    img = img[y1:y2, x1:x2]
    mask = mask[y1:y2, x1:x2]

    return img, mask, sitch_pitch_processing_method