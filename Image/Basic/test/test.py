import argparse
from tkinter import Y
import cv2
import numpy as np
import os


def absdiff(args):
    
    img_list = os.listdir(args.input_dir)
    img_list.sort()

    # init
    bkg_list = []
    size_thres = [300, 20000]

    # 生成椭圆结构元素
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 背景建模
    for idx in range(len(img_list)):
        img_name = img_list[idx]
        img_path = os.path.join(args.input_dir, img_name)

        if 'base' in img_name:
            img = cv2.imread(img_path)

            # 灰度图
            img_bkg= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 对灰度图进行高斯模糊,平滑图像
            img_bkg = cv2.GaussianBlur(img_bkg, (11, 11), 0)

            bkg_list.append(img_bkg)

    for idx in range(len(img_list)):

        img_name = img_list[idx]
        img_path = os.path.join(args.input_dir, img_name)

        if 'base' not in img_name:

            ori_img = cv2.imread(img_path)

            # 灰度图
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

            # 对灰度图进行高斯模糊,平滑图像
            img = cv2.GaussianBlur(img, (11, 11), 0)

            diff = np.zeros(img.shape)
            # 获取当前帧与背景帧之间的图像差异,得到差分图
            for idy in range(len(bkg_list)):
                bkg_np = bkg_list[idy].astype(np.uint8)
                img = img.astype(np.uint8)
                diff += 0.33 * cv2.absdiff(bkg_np, img)

            # 利用像素点值进行阈值分割,得到一副黑白图像
            diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

            # 膨胀图像,减少错误
            diff = cv2.dilate(diff, es, iterations=2)

            # 得到图像中的目标轮廓
            diff = diff.astype(np.uint8)
            cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                if cv2.contourArea(c) < size_thres[0] or cv2.contourArea(c) > size_thres[1]:
                    continue
                # 绘制目标矩形框
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(ori_img, (x+2, y+2), (x+w, y+h), (0, 255, 0), 2)

            output_img_path = os.path.join(args.output_dir, img_name)
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            cv2.imwrite(output_img_path, ori_img)


def harris(image):

    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04
    corner_thres = 100

    # Detecting corners
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > corner_thres:
                cv2.circle(image, (j, i), 2, (0, 255, 0), 2)

    return image


def shi_tomas(image):

    # Detector parameters
    axCorners = 5000
    qualityLevel = 0.05
    minDistance = 5

    # Detecting corners
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, axCorners, qualityLevel, minDistance)

    print(len(corners))
    for pt in corners:
        x = np.int32(pt[0][0])
        y = np.int32(pt[0][1])
        cv2.circle(image, (x, y), 2, (0, 255, 0), 2)

    return image


def shi_tomas_subpix(image):

    # Detector parameters
    axCorners = 5000
    qualityLevel = 0.05
    minDistance = 5

    # Detecting corners
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, axCorners, qualityLevel, minDistance)
    print(len(corners))
    for pt in corners:
        x = np.int32(pt[0][0])
        y = np.int32(pt[0][1])
        cv2.circle(image, (x, y), 2, (0, 255, 0), 2)

    # detect sub-pixel
    winSize = (3, 3)
    zeroZone = (-1, -1)

    # Stop condition
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
    # Calculate the refined corner locations
    corners = cv2.cornerSubPix(gray, corners, winSize, zeroZone, criteria)
    
    # # display
    # for i in range(corners.shape[0]):
    #     print(" -- Refined Corner [", i, "]  (", corners[i, 0, 0], ",", corners[i, 0, 1], ")")
    return image


def fast_corner(image, cnts_list=[]):
    
    # init
    res_list = []

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()
    # fast.setNonmaxSuppression(0)

    # find and draw the keypoints
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kp = fast.detect(img_gray, None)

    if len(cnts_list):
        for idx in range(len(cnts_list)):
            x1, y1, x2, y2 = cnts_list[idx]

            kp_dict = {}
            kp_dict['cnts'] = [x1, y1, x2, y2]
            kp_dict['kps'] = []

            for idy in range(len(kp)):
                kp_idy = kp[idy]
                if kp_idy.pt[0] >= x1 and \
                    kp_idy.pt[0] <= x2 and \
                    kp_idy.pt[1] >= y1 and \
                    kp_idy.pt[1] <= y2:

                    kp_dict['kps'].append([int(kp_idy.pt[0]), int(kp_idy.pt[1])])
            
            w = x2 - x1 
            y = y2 - y1
            size = w * y
            if len(kp_dict['kps']) > size * 0.02:
                res_list.append(kp_dict)
        
        return image, res_list
    else:
        image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0))

        return image
    

def corner(args):
    
    img_list = os.listdir(args.input_dir)
    img_list.sort()

    for idx in range(len(img_list)):   

        img_name = img_list[idx]
        img_path = os.path.join(args.input_dir, img_name)

        if 'base' not in img_name:

            ori_img = cv2.imread(img_path)

            # ori_img = harris(ori_img)
            # ori_img = shi_tomas(ori_img)
            # ori_img = shi_tomas_subpix(ori_img)
            ori_img = fast_corner(ori_img)

            output_img_path = os.path.join(args.output_dir, img_name)
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            cv2.imwrite(output_img_path, ori_img)


def absdiff_corner(args):

    img_list = os.listdir(args.input_dir)
    img_list.sort()

    # init
    bkg_list = []
    size_thres = [300, 6000]

    # 生成椭圆结构元素
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 背景建模
    for idx in range(len(img_list)):
        img_name = img_list[idx]
        img_path = os.path.join(args.input_dir, img_name)

        if 'base' in img_name:
            img = cv2.imread(img_path)

            # 灰度图
            img_bkg= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 对灰度图进行高斯模糊,平滑图像
            img_bkg = cv2.GaussianBlur(img_bkg, (11, 11), 0)

            bkg_list.append(img_bkg)

    for idx in range(len(img_list)):

        img_name = img_list[idx]
        img_path = os.path.join(args.input_dir, img_name)

        if 'base' not in img_name:

            ori_img = cv2.imread(img_path)

            # 灰度图
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

            # 对灰度图进行高斯模糊,平滑图像
            img = cv2.GaussianBlur(img, (11, 11), 0)

            diff = np.zeros(img.shape)
            # 获取当前帧与背景帧之间的图像差异,得到差分图
            for idy in range(len(bkg_list)):
                bkg_np = bkg_list[idy].astype(np.uint8)
                img = img.astype(np.uint8)
                diff += 0.33 * cv2.absdiff(bkg_np, img)

            # 利用像素点值进行阈值分割,得到一副黑白图像
            diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

            # 膨胀图像,减少错误
            diff = cv2.dilate(diff, es, iterations=2)

            # 得到图像中的目标轮廓
            diff = diff.astype(np.uint8)
            cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cnts_list = []
            for c in cnts:
                if cv2.contourArea(c) < size_thres[0] or cv2.contourArea(c) > size_thres[1]:
                    continue
                # 绘制目标矩形框
                (x, y, w, h) = cv2.boundingRect(c)
                cnts_list.append([x, y, w, h])
            
            # res_cnts_list = cnts_list
            _, res_cnts_list = rectMerge_sxf(cnts_list)
            cnts_list = []
            for c in res_cnts_list:
                x, y, w, h = c
                cnts_list.append([x, y, x+w, y+h])

            ori_img, res_list = fast_corner(ori_img, cnts_list)
            
            for idy in range(len(res_list)):
                x1, y1, x2, y2 = res_list[idy]['cnts']
                cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for idz in range(len(res_list[idy]['kps'])):
                    kps_idx = res_list[idy]['kps'][idz]
                    cv2.circle(ori_img, (int(kps_idx[0]), int(kps_idx[1])), 2, (0, 255, 0), 2)
                    
            output_img_path = os.path.join(args.output_dir, img_name)
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            cv2.imwrite(output_img_path, ori_img)


def checkOverlap(boxa, boxb):
    x1, y1, w1, h1 = boxa
    x2, y2, w2, h2 = boxb
    if (x1 > x2 + w2):
        return 0
    if (y1 > y2 + h2):
        return 0
    if (x1 + w1 < x2):
        return 0
    if (y1 + h1 < y2):
        return 0
    colInt = abs(min(x1 + w1, x2 + w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    return overlap_area / (area1 + area2 - overlap_area)
 

def unionBox(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return [x, y, w, h]
 
 
def intersectionBox(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return ()
    return [x, y, w, h]
 

def rectMerge_sxf(rects: []):
    '''
    当通过connectedComponentsWithStats找到rects坐标时，
    注意前2個坐标是表示整個圖的，需要去除，不然就只有一個大框，
    在执行此函数前，可执行类似下面的操作。
    rectList = sorted(rectList)[2:]
    '''
    # rects => [[x1, y1, w1, h1], [x2, y2, w2, h2], ...]
    rectList = rects.copy()
    rectList.sort()
    new_array = []
    complete = 1
    # 要用while，不能forEach，因爲rectList內容會變
    i = 0
    while i < len(rectList):
        # 選後面的即可，前面的已經判斷過了，不需要重復操作
        j = i + 1
        succees_once = 0
        while j < len(rectList):
            boxa = rectList[i]
            boxb = rectList[j]
            # 判斷是否有重疊，注意只針對水平＋垂直情況，有角度旋轉的不行
            if checkOverlap(boxa, boxb):  # intersectionBox(boxa, boxb)
                complete = 0
                # 將合並後的矩陣加入候選區
                new_array.append(unionBox(boxa, boxb))
                succees_once = 1
                # 從原列表中刪除，因爲這兩個已經合並了，不刪除會導致重復計算
                rectList.remove(boxa)
                rectList.remove(boxb)
                break
            j += 1
        if succees_once:
            # 成功合並了一次，此時i不需要+1，因爲上面進行了remove(boxb)操作
            continue
        i += 1
    # 剩餘項是不重疊的，直接加進來即可
    new_array.extend(rectList)
 
    # 0: 可能還有未合並的，遞歸調用;
    # 1: 本次沒有合並項，說明全部是分開的，可以結束退出
    if complete == 0:
        complete, new_array = rectMerge_sxf(new_array)
    return complete, new_array


def test(args):

    # absdiff(args)
    # corner(args)
    absdiff_corner(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_img/"
    args.output_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_res/"

    test(args)