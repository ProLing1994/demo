import cv2
import numpy as np
import os
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Demo.Diff_Cutout.utils.image_processing import *


class DiffCutoutAPI():
    """
    DiffCutoutAPI
    """

    def __init__(self):
        """
        初始化
        """

        # option
        self.option_init()

        # load bkg
        self.bkg_list = []

        # 生成椭圆结构元素
        self.es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


    def option_init(self):

        # 抠图方法
        self.diff_cutout_method = "gradient_sobel"

        # 大小阈值（滤除小框）
        self.size_thres = [200, 70000]


    def load_bkg(self, bkg_dir):

        img_list = os.listdir(bkg_dir)
        img_list.sort()

        # 背景建模
        for idx in range(len(img_list)):
            img_name = img_list[idx]
            img_path = os.path.join(bkg_dir, img_name)

            img = cv2.imread(img_path)

            # 灰度图
            img_bkg= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 对灰度图进行高斯模糊,平滑图像
            img_bkg = cv2.GaussianBlur(img_bkg, (11, 11), 0)

            if self.diff_cutout_method  == "gradient_sobel":
                img_bkg = sobel(img_bkg)
            else:
                raise Exception

            self.bkg_list.append(img_bkg)
    

    def run(self, img_path):
        """
        return:
        img
        bbox_list
        """                    
        # init 
        bbox_list = []

        ori_img = cv2.imread(img_path)

        # 灰度图
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

        # 对灰度图进行高斯模糊,平滑图像
        img = cv2.GaussianBlur(img, (11, 11), 0)

        if self.diff_cutout_method  == "gradient_sobel":
            img = sobel(img)
        else:
            raise Exception
        
        # 获取当前帧与背景帧之间的图像差异，得到差分图
        diff = np.zeros(img.shape)
        for idy in range(len(self.bkg_list)):
            bkg_np = self.bkg_list[idy]
            diff += (1/len(self.bkg_list)) * cv2.absdiff(bkg_np, img)
        
        # 利用像素点值进行阈值分割,得到一副黑白图像
        diff = cv2.threshold(diff, 70, 255, cv2.THRESH_BINARY)[1]
        # diff = diff.astype(np.uint8)
        # retval, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_OTSU)

        # 膨胀图像,减少错误
        diff = cv2.dilate(diff, self.es, iterations=6)

        # 得到图像中的目标轮廓
        diff = diff.astype(np.uint8)
        cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) < self.size_thres[0] or cv2.contourArea(c) > self.size_thres[1]:
                continue
            
            # 绘制目标矩形框
            (x, y, w, h) = cv2.boundingRect(c)
            bbox_list.append([x, y, x+w, y+h])

        return ori_img, bbox_list