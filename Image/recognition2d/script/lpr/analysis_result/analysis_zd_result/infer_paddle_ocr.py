import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys 
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from script.paddle.infer.lpr import LPRPaddle, LPROnnx, LPRCaffe


def model_test(args, from_jpg_dir=False, bool_onnx=False, bool_caffe=False):

    # mkdir 
    create_folder(os.path.dirname(args.output_csv_path))

    # lpr
    if bool_onnx:
        lpr = LPROnnx(args.config_path, args.model_path, padding_bool=False)
    elif bool_caffe:
        lpr = LPRCaffe(args.model_path, args.prototxt_path, args.dict_path, padding_bool=False)
    else:
        lpr = LPRPaddle(args.config_path, args.model_path)

    # img list
    img_list = []

    if from_jpg_dir==True:
        img_list = os.listdir(args.input_jpg_path)
        img_list = [os.path.join(args.input_jpg_path, img_name) for img_name in img_list]
    else:
        with open(args.img_list) as f:
            for line in f:
                img_list.append(line.strip())  

    # results list  
    results_list = []

    for idx in tqdm(range(len(img_list))):
        # init 
        results_dict = {}

        img_name = os.path.basename(img_list[idx])
        img_path = img_list[idx]
        tqdm.write(img_path)

        if bool_onnx or bool_caffe:
            img = cv2.imread(img_path)
        else:
            with open(img_path, 'rb') as f:
                img = f.read()

        ocr, ocr_score = lpr.run(img)

        # pd 
        results_dict['file'] = img_path
        results_dict['label'] = img_name.replace('.jpg', '').split('_')[-1].replace(' - 副本', '')
        
        results_dict['ocr'] = ocr
        results_dict['ocr_score'] = np.array(ocr_score).mean()
        results_dict['res'] = int( results_dict['label'] == results_dict['ocr'] )
        # results_dict['res'] = int( results_dict['label'][1:] == results_dict['ocr'][1:] )

        tqdm_write = '{} {} {}'.format( results_dict['label'], results_dict['ocr'], results_dict['res'] )
        tqdm.write(tqdm_write)

        results_list.append(results_dict)

    # out csv
    csv_data_pd = pd.DataFrame(results_list)
    csv_data_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # ###############################
    # # paddle
    # ###############################
    # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_20230119_all_aug/config.yml"
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_20230119_all_aug/best_accuracy"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_20230119_all_aug/best_accuracy/"

    # # ocr_merge_test
    # args.img_list = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_202301/ImageSetsNoAug/ImageSets/Main/test.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test/ocr_merge_test_result.csv')
    # model_test(args)


    # ###############################
    # # onnx
    # ###############################

    # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_1219_all_aug/config.yml"
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_1219_all_aug/inference/onnx/model.onnx"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_1219_all_aug/best_accuracy/"

    # # ocr_merge_test
    # args.img_list = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/ocr_merge_test/ImageSets/Main/test.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_onnx/ocr_merge_test_result.csv')
    # model_test(args, bool_onnx=True)

    # # # # from_jpg_dir
    # # args.input_jpg_path = "/yuanhuan/data/image/ZD_anpr/test_video/ZD_DUBAI/jpg文件/特殊车牌_crop/豹子号"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_onnx/data_synthesis_baozihao_result.csv')
    # # model_test(args, from_jpg_dir=True, bool_onnx=True)


    ###############################
    # caffe
    ###############################

    args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707/inference/caffe/model.caffemodel"
    args.prototxt_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707/inference/caffe/model.prototxt"
    args.dict_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707/inference/cn_dict.txt"
    args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707/best_accuracy/"

    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707_original_248/inference/caffe/model.caffemodel"
    # args.prototxt_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707_original_248/inference/caffe/model.prototxt"
    # args.dict_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707_original_248/inference/cn_dict.txt"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_chn_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230707_original_248/best_accuracy/"

    # ocr_merge_test
    # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_cn_202305/sichuan/ImageSets/Main/train.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_sichuan_train_result.csv')
    # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_cn_202305/sichuan/ImageSets/Main/test.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_sichuan_test_result.csv')
    # model_test(args, bool_caffe=True)

    # from_jpg_dir
    # args.input_jpg_path = "/yuanhuan/data/image/RM_ANPR/original/cn/DIFFSTE/original_248_52/train"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_original_248_52_train_result.csv')
    # args.input_jpg_path = "/yuanhuan/data/image/RM_ANPR/original/cn/DIFFSTE/original_248_52/val"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_original_248_52_val_result.csv')
    args.input_jpg_path = "/yuanhuan/data/image/RM_ANPR/training/plate_cn_202305/sichuan/Images/"
    args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_sichuan_result.csv')
    # args.input_jpg_path = "/yuanhuan/data/image/RM_ANPR/training/plate_cn_202305/sichuan/Images/"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_sichuan_result_no_char.csv')
    model_test(args, from_jpg_dir=True, bool_caffe=True)
