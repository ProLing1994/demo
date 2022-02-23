import argparse
import cv2
import numpy as np
import os
import sys 
import torch
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
from regreesion2d.plate_regreesion.infer.ssd_vgg_fpn import SSDDetector
from regreesion2d.plate_regreesion.infer.plate_regression import PlateRegression
from regreesion2d.plate_regreesion.utils.draw_tools import draw_detection_result

from recognition2d.license_plate_recognition.infer.license_plate import license_palte_model_init_caffe, license_palte_crnn_recognition_caffe, license_palte_beamsearch_init, license_palte_crnn_recognition_beamsearch_caffe


def inference_images(args):
    # mkdir 
    if args.write_bool:
        if not os.path.isdir(args.output_img_dir):
            os.makedirs(args.output_img_dir)

    # model init
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    car_detector = SSDDetector(device=device, weight_path=args.car_model_path)
    plate_detector = PlateRegression(args.plate_model_path, args.plate_config_path, device)
    license_palte_detector = license_palte_model_init_caffe(args.plate_regression_prototxt, args.plate_regression_model_path)

    # image init 
    img_list = np.array(os.listdir(args.img_dir))
    img_list = img_list[[jpg.endswith('.jpg') for jpg in img_list]]
    img_list.sort()
    
    for idx in tqdm(range(len(img_list))):
        img_path = os.path.join(args.img_dir, img_list[idx])

        if args.write_bool:
            output_img_path = os.path.join(args.output_img_dir, img_list[idx])
        
        tqdm.write(img_path)

        img = cv2.imread(img_path)
        plate_img = cv2.imread(img_path, 0)

        # init
        bboxes = {}
        show_bboxes = {}

        # car_detector
        bboxes['car'] = car_detector.detect(img)['car']

        # plate_detector
        bboxes["license_plate"] = plate_detector.detect(img, bboxes['car'])
        

        for plate_idx in range(len(bboxes["license_plate"])):
            plate_bbox = bboxes["license_plate"][plate_idx]

            crop_img = plate_img[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]
            result_ocr, result_scors_list = license_palte_crnn_recognition_caffe(license_palte_detector, crop_img)

            if args.height_threshold_bool and args.ocr_threshold_bool:
                # 方式一：高度阈值判断，ocr 阈值判断
                plate_height = plate_bbox[3] - plate_bbox[1]
                if plate_height >= args.height_threshold:
                    if np.array(result_scors_list).mean() >= args.ocr_threshold:
                        show_bboxes[result_ocr] = [plate_bbox]
            elif args.height_threshold_bool and not args.ocr_threshold_bool:
                # 方式二：高度阈值判断
                plate_height = plate_bbox[3] - plate_bbox[1]
                if plate_height >= args.height_threshold:
                    show_bboxes[result_ocr] = [plate_bbox]
            elif not args.height_threshold_bool and args.ocr_threshold_bool:
                # 方式三：ocr 阈值判断
                if np.array(result_scors_list).mean() >= args.ocr_threshold:
                    show_bboxes[result_ocr] = [plate_bbox]
            else:
                # 方式四：直接叠加
                show_bboxes[result_ocr] = [plate_bbox]                   

        # draw img
        if args.write_bool:
            # img = draw_detection_result(img, bboxes, mode='ltrb')
            img = draw_detection_result(img, show_bboxes, mode='ltrb')
            cv2.imwrite(output_img_path, img)


def inference_vidio(args):
    # mkdir 
    if args.write_bool:
        if not os.path.isdir(args.output_vidio_dir):
            os.makedirs(args.output_vidio_dir)

    # model init
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    car_detector = SSDDetector(device=device, weight_path=args.car_model_path)
    plate_detector = PlateRegression(args.plate_model_path, args.plate_config_path, device)
    license_palte_detector = license_palte_model_init_caffe(args.plate_regression_prototxt, args.plate_regression_model_path)
    license_palte_beamsearch = license_palte_beamsearch_init()

    # image init 
    vidio_list = np.array(os.listdir(args.vidio_dir))
    vidio_list = vidio_list[[vidio.endswith('.avi') for vidio in vidio_list]]
    vidio_list.sort()
    
    for idx in tqdm(range(len(vidio_list))):
        vidio_path = os.path.join(args.vidio_dir, vidio_list[idx])

        if args.write_bool:
            output_vidio_path = os.path.join(args.output_vidio_dir, vidio_list[idx])
        
        cap = cv2.VideoCapture(vidio_path) 
        print(int(cap.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧

        if args.write_bool:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            video_writer = cv2.VideoWriter(output_vidio_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        frame_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret: # if the camera over return false
                video_writer.release()
                break
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # init
            bboxes = {}
            show_bboxes = {}

            # car_detector
            bboxes['car'] = car_detector.detect(frame)['car']

            # plate_detector
            bboxes["license_plate"] = plate_detector.detect(frame, bboxes['car'])
            

            for plate_idx in range(len(bboxes["license_plate"])):
                plate_bbox = bboxes["license_plate"][plate_idx]

                crop_img = frame_gray[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]
                # check
                if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                    continue

                if args.prefix_beam_search_bool:
                    # prefix beamsearch
                    _, result_scors_list = license_palte_crnn_recognition_caffe(license_palte_detector, crop_img)
                    result_ocr = license_palte_crnn_recognition_beamsearch_caffe(license_palte_detector, crop_img, license_palte_beamsearch[0], license_palte_beamsearch[1])
                else:
                    # greedy
                    result_ocr, result_scors_list = license_palte_crnn_recognition_caffe(license_palte_detector, crop_img)

                if args.height_threshold_bool and args.ocr_threshold_bool:
                    # 方式一：高度阈值判断，ocr 阈值判断
                    plate_height = plate_bbox[3] - plate_bbox[1]
                    if plate_height >= args.height_threshold:
                        if np.array(result_scors_list).mean() >= args.ocr_threshold:
                            show_bboxes[result_ocr] = [plate_bbox]
                elif args.height_threshold_bool and not args.ocr_threshold_bool:
                    # 方式二：高度阈值判断
                    plate_height = plate_bbox[3] - plate_bbox[1]
                    if plate_height >= args.height_threshold:
                        show_bboxes[result_ocr] = [plate_bbox]
                elif not args.height_threshold_bool and args.ocr_threshold_bool:
                    # 方式三：ocr 阈值判断
                    if np.array(result_scors_list).mean() >= args.ocr_threshold:
                        show_bboxes[result_ocr] = [plate_bbox]
                else:
                    # 方式四：直接叠加
                    show_bboxes[result_ocr] = [plate_bbox]                       

            # draw img
            if args.write_bool:
                frame = draw_detection_result(frame, show_bboxes, mode='ltrb')
                video_writer.write(frame)

                output_img_path = os.path.join(args.output_vidio_dir, vidio_list[idx].replace('.avi', '_{}.jpg'.format(frame_idx)))
                cv2.imwrite(output_img_path, frame)
                frame_idx += 1

                tqdm.write("{}: {}".format(vidio_path, str(frame_idx)))


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.car_model_path = "/home/huanyuan/code/demo/Image/regreesion2d/plate_regreesion/network/ssd_detector/SSD_VGG_FPN_VOC_epoches_165.pth"

    # args.plate_model_path = "/mnt/huanyuan2/model/image_model/license_plate_regressioin_model_wjh/国内、新加坡/MobileNetSmallV1_PO_singapore_Wdm_2020_03_19_17_36_30_PT_best/MobileNetSmallV1_PO_singapore_Wdm_2020_03_19_17_36_30_PT_best.pt"
    # args.plate_config_path = "/home/huanyuan/code/demo/Image/regreesion2d/plate_regreesion/config/config.py"
    args.plate_model_path = "/mnt/huanyuan2/model/image_model/license_plate_regressioin_model_lpj/best.pt"
    args.plate_config_path = "/mnt/huanyuan2/model/image_model/license_plate_regressioin_model_lpj/config.py"

    args.plate_regression_prototxt = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china_softmax.prototxt"
    args.plate_regression_model_path = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china.caffemodel"
    args.prefix_beam_search_bool = True

    args.write_bool = True
    args.height_threshold_bool = True
    args.height_threshold = 24
    args.ocr_threshold_bool = True
    args.ocr_threshold = 0.8
    
    args.img_bool = False
    args.img_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试样本/2MH/"
    # args.output_img_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试样本_wo_plate_thres/2MH/"
    args.output_img_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试样本_plate_thres/2MH/"

    if args.img_bool:
        inference_images(args)

    args.vidio_bool = True
    args.vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频/测试视频/"
    # args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频_wo_height_thres_wo_plate_thres/"
    # args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频_height_thres_wo_plate_thres/"
    # args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频_height_thres_plate_thres/"
    args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频_height_thres_plate_thres_prefix_beam_search/"

    if args.vidio_bool:
        inference_vidio(args)


if __name__ == '__main__':
    main()