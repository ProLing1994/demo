import argparse
import cv2
import numpy as np
import os
import sys 
import torch
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
# sys.path.insert(0, '/yuanhuan/code/demo/Image')
from regreesion2d.plate_regreesion.infer.ssd_vgg_fpn import SSDDetector
from regreesion2d.plate_regreesion.infer.plate_regression import PlateRegression
from regreesion2d.plate_regreesion.utils.draw_tools import draw_detection_result


def inference_images(args):
    # mkdir 
    if args.write_img_bool:
        if not os.path.isdir(args.output_img_dir):
            os.makedirs(args.output_img_dir)

    # model init
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    car_detector = SSDDetector(device=device, weight_path=args.car_model_path)
    plate_detector = PlateRegression(args.plate_model_path, args.plate_config_path, device)

    # image init 
    img_list = np.array(os.listdir(args.img_dir))
    img_list = img_list[[jpg.endswith('.jpg') for jpg in img_list]]
    img_list.sort()
    
    for idx in tqdm(range(len(img_list))):
        img_path = os.path.join(args.img_dir, img_list[idx])

        if args.write_img_bool:
            output_img_path = os.path.join(args.output_img_dir, img_list[idx])
        
        tqdm.write(img_path)

        img = cv2.imread(img_path)
        
        # init
        bboxes = {}
        # car_detector
        bboxes['car'] = car_detector.detect(img)['car']

        # plate_detector
        bboxes["license_plate"] = plate_detector.detect(img, bboxes['car'])
        
        # draw img
        bboxes['car'] = []
        if args.write_img_bool:
            img = draw_detection_result(img, bboxes, mode='ltrb')
            cv2.imwrite(output_img_path, img)


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.plate_model_path = "/mnt/huanyuan2/model/image_model/license_plate_regressioin_model_wjh/国内、新加坡/MobileNetSmallV1_PO_singapore_Wdm_2020_03_19_17_36_30_PT_best/MobileNetSmallV1_PO_singapore_Wdm_2020_03_19_17_36_30_PT_best.pt"
    args.plate_config_path = "/home/huanyuan/code/demo/Image/regreesion2d/plate_regreesion/config/config.py"
    args.car_model_path = "/home/huanyuan/code/demo/Image/regreesion2d/plate_regreesion/network/ssd_detector/SSD_VGG_FPN_VOC_epoches_165.pth"
    args.img_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试样本/2MH/"
    args.output_img_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试样本_test/2MH/"
    args.write_img_bool = True

    inference_images(args)


if __name__ == '__main__':
    main()