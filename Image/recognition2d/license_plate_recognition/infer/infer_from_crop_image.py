import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
# sys.path.insert(0, '/yuanhuan/code/demo/Image')
from recognition2d.license_plate_recognition.infer.license_plate import license_palte_model_init_caffe, license_palte_crnn_recognition_caffe, license_palte_beamsearch_init, license_palte_crnn_recognition_beamsearch_caffe

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *


def model_init(args):
    # model init
    license_palte_detector = license_palte_model_init_caffe(args.plate_recognition_prototxt, args.plate_recognition_model_path)
    license_palte_beamsearch = license_palte_beamsearch_init()
    return (license_palte_detector, license_palte_beamsearch)


def img_detect(args, model, img):
    license_palte_detector = model[0]
    license_palte_beamsearch = model[1]

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.prefix_beam_search_bool:
        # prefix beamsearch
        _, result_scors_list = license_palte_crnn_recognition_caffe(license_palte_detector, gray_img)
        result_ocr = license_palte_crnn_recognition_beamsearch_caffe(license_palte_detector, gray_img, license_palte_beamsearch[0], license_palte_beamsearch[1])
    else:
        # greedy
        result_ocr, result_scors_list = license_palte_crnn_recognition_caffe(license_palte_detector, gray_img)           

    return result_ocr, np.array(result_scors_list).mean()


def inference_images(args):
    # model init
    model = model_init(args)

    # image init 
    img_list = np.array(os.listdir(args.img_dir))
    img_list = img_list[[jpg.endswith('.jpg') for jpg in img_list]]
    img_list.sort()
    
    # init 
    results_list = []

    for idx in tqdm(range(len(img_list))):
        # init 
        results_dict = {}

        img_path = os.path.join(args.img_dir, img_list[idx])
        tqdm.write(img_path)

        img = cv2.imread(img_path)        

        # info 
        image_width = img.shape[1]
        image_height = img.shape[0]

        # detect 
        ocr, ocr_score = img_detect(args, model, img)

        # pd 
        results_dict['file'] = img_list[idx]
        # results_dict['label'] = str(img_list[idx]).split('@')[1].split('-')[0]
        results_dict['label'] = str(img_list[idx]).split('_')[0]
        results_dict['ocr'] = ocr
        results_dict['ocr_score'] = ocr_score
        results_dict['res'] = int( results_dict['label'][1:] == results_dict['ocr'][1:] )
        results_dict['width'] = image_width
        results_dict['height'] = image_height
        
        tqdm_write = '{} {} {}'.format( results_dict['label'], results_dict['ocr'], int( results_dict['label'] == results_dict['ocr'] ) )
        tqdm.write(tqdm_write)

        results_list.append(results_dict)

    # out csv
    csv_data_pd = pd.DataFrame(results_list)
    csv_data_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")

def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.plate_recognition_prototxt = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china_softmax.prototxt"
    args.plate_recognition_model_path = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china.caffemodel"
    args.prefix_beam_search_bool = False
    
    # normal，林旭南提供数据
    # args.name_format = "blue"
    # # args.name_format = "green"
    # args.img_dir = os.path.join("/mnt/huanyuan2/data/image/LicensePlateRecognition/normal", args.name_format)
    # args.output_csv_path = os.path.join("/mnt/huanyuan2/data/image/LicensePlateRecognition/normal", args.name_format + ".csv")
    
    # # zg，智观加油站数据 2M
    # # args.name_format = "blue"
    # # args.name_format = "green"
    # args.img_dir = os.path.join("/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_zhjyz_2M", args.name_format)
    # args.output_csv_path = os.path.join("/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_zhjyz_2M", args.name_format + "_zi.csv")

    # zg，安徽淮北高速 5M
    # args.name_format = "blue"
    # args.name_format = "green"
    args.name_format = "temp"
    args.img_dir = os.path.join("/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_ahhbgs_5M", args.name_format)
    args.output_csv_path = os.path.join("/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_ahhbgs_5M", args.name_format + ".csv")

    inference_images(args)


if __name__ == '__main__':
    main()