import cv2
import numpy as np


image_shape = [3, 1920, 2592]
output_img_path = "/mnt/huanyuan/model_final/image_model/lpr_zd/white.jpg"

imgC, imgH, imgW = image_shape
img = np.ones((imgH, imgW, imgC), dtype=np.uint8)
img *= 255

cv2.imwrite(output_img_path, img)