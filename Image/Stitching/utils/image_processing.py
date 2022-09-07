import cv2
import numpy as np
import random


def otsu(img, erode_iter=1, dilate_iter=4):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_OTSU)

    mask = cv2.erode(mask, None, iterations=erode_iter)
    mask = cv2.dilate(mask, None, iterations=dilate_iter)

    mask = np.expand_dims(mask, axis=2)

    return img * mask, mask


def gaussian_otsu(img, erode_iter=0, dilate_iter=4):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    retval, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_OTSU)

    mask = cv2.erode(mask, None, iterations=erode_iter)
    mask = cv2.dilate(mask, None, iterations=dilate_iter)

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
