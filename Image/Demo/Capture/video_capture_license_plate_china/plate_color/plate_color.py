import argparse
import cv2
import numpy as np
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *


# 设定阈值
# 阈值设置参考：https://blog.csdn.net/wanggsx918/article/details/23272669
lower_blue = np.array([90, 43, 46])
upper_blue = np.array([124, 255, 255])

lower_green = np.array([35, 43, 46])
upper_green = np.array([89, 255, 255])

lower_yellow = np.array([11, 43, 46])
upper_yellow = np.array([34, 255, 255])

lower_white = np.array([0, 0, 221])
upper_white = np.array([0, 30, 255])


def update_plate_color(img):

    # 转换为HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 根据阈值构建掩膜
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # 将掩膜灰度图转化为2值图
    mask_blue[mask_blue==255] = 1
    mask_yellow[mask_yellow==255] = 1
    mask_green[mask_green==255] = 1
    mask_white[mask_white==255] = 1
    
    # color_list = ['blue', 'yellow', 'green']
    # num_list = [mask_blue.sum(), mask_yellow.sum(), mask_green.sum()]
    color_list = ['blue', 'yellow', 'green', 'white']
    num_list = [mask_blue.sum(), mask_yellow.sum(), mask_green.sum(), mask_white.sum()]

    img_color = color_list[num_list.index(max(num_list))]

    return img_color


def main():
    
    # 车牌颜色分析
    # 参考博客：https://blog.csdn.net/Aiden_yan/article/details/118459034
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # # blue
    # args.image_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/normal/blue_height_25_30"
    # args.image_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_zhjyz_2M/blue"
    args.image_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_ahhbgs_5M/blue"
    args.suffix = '.jpg'
    args.default_color = 'blue'

    # # green
    # args.image_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/normal/green"
    # # args.image_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_zhjyz_2M/green"
    # args.suffix = '.jpg'
    # args.default_color = 'green'

    # # yellow
    # args.image_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_ahhbgs_5M/yellow"
    # args.suffix = '.jpg'
    # args.default_color = 'yellow'

    # # white
    # args.image_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_ahhbgs_5M/white"
    # args.suffix = '.jpg'
    # args.default_color = 'white'

    # image init 
    img_list = np.array(os.listdir(args.image_dir))
    img_list = img_list[[img.endswith(args.suffix) for img in img_list]]
    img_list.sort()

    # init
    tp = 0
    fn = 0
    fn_list = []
    for idx in tqdm(range(len(img_list))):
        image_path = os.path.join(args.image_dir, img_list[idx])

        img = cv2.imread(image_path)
        img_color = update_plate_color(img)

        if img_color == args.default_color:
            tp += 1
        else:
            fn += 1 
            fn_list.append([img_list[idx], img_color])
        
        print('{}: {}'.format(img_list[idx], img_color))
    
    print("tpr: {:.2f}({}/{})".format(tp/(tp+fn), tp, tp+fn))
    print(fn_list)


if __name__ == '__main__':

    main()