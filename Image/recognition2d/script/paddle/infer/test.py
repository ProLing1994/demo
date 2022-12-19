import cv2
import numpy as np


image_shape = [3, 1920, 2592]
output_img_path = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/type/black.jpg"

imgC, imgH, imgW = image_shape
img = np.zeros((imgH, imgW, imgC), dtype=np.uint8)

cv2.imwrite(output_img_path, img)