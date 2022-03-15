import argparse
import cv2
import numpy as np
import os
import sys 
import torch
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
# sys.path.insert(0, '/yuanhuan/code/demo/Image')
from detection2d.ssd_rfb_crossdatatraining.test_tools import SSDDetector
from regreesion2d.plate_regreesion.utils.draw_tools import draw_detection_result

from recognition2d.license_plate_recognition.infer.license_plate import license_palte_model_init_caffe, license_palte_crnn_recognition_caffe, license_palte_beamsearch_init, license_palte_crnn_recognition_beamsearch_caffe


def model_init(args):
    # model init
    car_plate_detector = SSDDetector(prototxt=args.ssd_car_plate_prototxt, model_path=args.ssd_car_plate_model_path, ssd_caffe_bool=args.ssd_caffe_bool, merge_class_bool=args.merge_class_bool)
    license_palte_detector = license_palte_model_init_caffe(args.plate_recognition_prototxt, args.plate_recognition_model_path)
    license_palte_beamsearch = license_palte_beamsearch_init()
    return (car_plate_detector, license_palte_detector, license_palte_beamsearch)


def img_detect(args, model, img):
    car_plate_detector = model[0]
    license_palte_detector = model[1]
    license_palte_beamsearch = model[2]

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # init
    show_bboxes = {}

    # car_plate_detector
    bboxes = car_plate_detector.detect(img)

    # roi ignore 
    if args.roi_ignore_bool:
        # init
        temp_bboxes = bboxes
        bboxes = dict()

        for key in temp_bboxes.keys():
            bbox_list = []
            for bbox_idx in range(len(temp_bboxes[key])):
                bbox = temp_bboxes[key][bbox_idx]

                if bbox[2] < args.roi_ignore_area[0] or \
                    bbox[3] < args.roi_ignore_area[1] or \
                    bbox[0] > args.roi_ignore_area[2] or \
                    bbox[1] > args.roi_ignore_area[3]:
                    pass
                else:
                    bbox_list.append(bbox)

            bboxes[key] = bbox_list

    for key in bboxes.keys():
        if key != "license_plate":
            show_bboxes[key] = bboxes[key]

    for plate_idx in range(len(bboxes["license_plate"])):
        plate_bbox = bboxes["license_plate"][plate_idx]

        # plate_bbox_expand
        if args.plate_bbox_expand_bool:
            plate_height = plate_bbox[3] - plate_bbox[1]
            if plate_height >= args.height_threshold:
                pass
            elif plate_height >= args.plate_bbox_minist_height:
                expand_height = int(( args.height_threshold - plate_height) / 2.0 + 0.5 )
                plate_bbox[1] = max(0, plate_bbox[1] - expand_height)
                plate_bbox[3] = min(img.shape[0], plate_bbox[3] + expand_height)

        # crop
        crop_img = gray_img[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]

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
                else:
                    show_bboxes["ocr_ignore_plate"] = [plate_bbox]
            else:
                show_bboxes["height_ignore_plate"] = [plate_bbox]

        elif args.height_threshold_bool and not args.ocr_threshold_bool:
            # 方式二：高度阈值判断
            plate_height = plate_bbox[3] - plate_bbox[1]
            if plate_height >= args.height_threshold:
                show_bboxes[result_ocr] = [plate_bbox]
            else:
                show_bboxes["height_ignore_plate"] = [plate_bbox]
        elif not args.height_threshold_bool and args.ocr_threshold_bool:
            # 方式三：ocr 阈值判断
            if np.array(result_scors_list).mean() >= args.ocr_threshold:
                show_bboxes[result_ocr] = [plate_bbox]
            else:
                show_bboxes["ocr_ignore_plate"] = [plate_bbox]
        else:
            # 方式四：直接叠加
            show_bboxes[result_ocr] = [plate_bbox]                   

    return show_bboxes


def inference_images(args):
    # mkdir 
    if args.write_bool:
        if not os.path.isdir(args.output_img_dir):
            os.makedirs(args.output_img_dir)

    # model init
    model = model_init(args)

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

        # detect 
        show_bboxes = img_detect(args, model, img)

        # draw img
        if args.write_bool:
            img = draw_detection_result(img, show_bboxes, mode='ltrb')
            cv2.imwrite(output_img_path, img)


def inference_vidio(args):
    # mkdir 
    if args.write_bool:
        if not os.path.isdir(args.output_vidio_dir):
            os.makedirs(args.output_vidio_dir)

    # model init
    model = model_init(args)

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
            ret, img = cap.read()

            if not ret: # if the camera over return false
                video_writer.release()
                break

            # detect 
            show_bboxes = img_detect(args, model, img)

            # draw img
            if args.write_bool:
                img = draw_detection_result(img, show_bboxes, mode='ltrb')
                video_writer.write(img)

                output_img_path = os.path.join(args.output_vidio_dir, vidio_list[idx].replace('.avi', '_{}.jpg'.format(frame_idx)))
                cv2.imwrite(output_img_path, img)
                frame_idx += 1

                tqdm.write("{}: {}".format(vidio_path, str(frame_idx)))


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.ssd_car_plate_prototxt = None
    # args.ssd_car_plate_model_path = "/mnt/huanyuan/model/image/ssd_rfb/SSD_VGG_FPN_RFB_2022-02-24-15_focalloss_4class_car_bus_truck_licenseplate_zg_w_fuzzy_plate/SSD_VGG_FPN_RFB_VOC_epoches_299.pth"
    args.ssd_car_plate_prototxt = "/mnt/huanyuan2/model/image_model/ssd_rfb_zg/FPN_RFB_5class_noDilation_prior.prototxt"
    args.ssd_car_plate_model_path = "/mnt/huanyuan2/model/image_model/ssd_rfb_zg/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_zg_2022-02-24-15.caffemodel"
    args.ssd_caffe_bool = True

    args.plate_recognition_prototxt = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china_softmax.prototxt"
    args.plate_recognition_model_path = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china.caffemodel"
    args.prefix_beam_search_bool = True

    # 是否保存结果
    args.write_bool = True

    # 是否通过 roi 区域屏蔽部分检测结果
    args.roi_ignore_bool = False
    args.roi_ignore_area = [0, 100, 1920, 980]

    # 是否将 car\bus\truck 合并为一类输出
    args.merge_class_bool = True

    # 是否扩展高度不足 24 车牌
    args.plate_bbox_expand_bool = False
    args.plate_bbox_minist_height = 18

    # 是否设置高度阈值挑选车牌
    args.height_threshold_bool = False
    args.height_threshold = 24

    # 是否设置 ocr 阈值挑选车牌
    args.ocr_threshold_bool = False
    args.ocr_threshold = 0.8
    
    args.img_bool = False
    args.img_dir = "/mnt/huanyuan2/model/image_model/ssd_rfb_zg/"
    args.output_img_dir = "/mnt/huanyuan2/model/image_model/ssd_rfb_zg/test/"

    if args.img_bool:
        inference_images(args)

    args.vidio_bool = True
    # args.vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频/测试视频/"
    # # args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频_height_ocr_beamsearch_mergeclass_bboxexpand/"
    # # args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频_height_beamsearch_mergeclass_bboxexpand/"
    # # args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频_height_beamsearch_mergeclass_bboxexpand_roiignore/"
    # # args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频_height_beamsearchs_bboxexpand_roiignore/"
    # args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/加油站测试视频_beamsearchs_roiignore/"

    # args.vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/test/视频_20220310/264漏报视频/"
    # args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/test/视频_20220310/264漏报视频_beamsearchs/"

    # args.vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/test/视频_20220310/264误报视频/"
    # args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/test/视频_20220310/264误报视频_beamsearchs/"

    args.vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/test/test/"
    args.output_vidio_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/test/test_beamsearchs_caffe/"

    if args.vidio_bool:
        inference_vidio(args)


if __name__ == '__main__':
    main()