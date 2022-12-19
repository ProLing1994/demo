import cv2
import numpy as np 
import os

if __name__ == "__main__":

    # 比较 PC cv resiz 和 板端 cv resize 函数结果
    caffe_img_path = "/mnt/huanyuan2/data/image/HY_Tanker/算法误报_1121/jpg/side/7118000000000000-221116-225400-225420-01p013000000/7118000000000000-221116-225400-225420-01p013000000_00000.jpg"
    novt_img_path = "/mnt/huanyuan2/data/image/HY_Tanker/novt_resize/7118000000000000-221116-225400-225420-01p013000000_00000_resize.jpg"
    size = (256, 144)

    caffe_img = cv2.imread(caffe_img_path)
    caffe_img = cv2.resize(caffe_img, size)

    novt_img = cv2.imread(novt_img_path)

    diff_output = caffe_img - novt_img
    output_path = novt_img_path[:-4] + '_diff_img.jpg'
    cv2.imwrite(output_path, diff_output)