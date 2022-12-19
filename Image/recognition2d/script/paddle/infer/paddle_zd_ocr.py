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
from script.paddle.infer.lpy import LPRPaddle, LPROnnx, LPRCaffe


def model_test(args, from_jpg_dir=False, bool_onnx=False, bool_caffe=False):

    # mkdir 
    create_folder(os.path.dirname(args.output_csv_path))

    # lpr
    if bool_onnx:
        lpr = LPROnnx(args.config_path, args.model_path)
    elif bool_caffe:
        lpr = LPRCaffe(args.model_path, args.prototxt_path, args.dict_path)
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

        results_dict['label'] = img_name.replace('.jpg', '').split('_')[-1]
        results_dict['ocr'] = ocr
        results_dict['ocr_score'] = np.array(ocr_score).mean()
        results_dict['res'] = int( results_dict['label'] == results_dict['ocr'] )

        tqdm_write = '{} {} {}'.format( results_dict['label'], results_dict['ocr'], int( results_dict['label'] == results_dict['ocr'] ) )
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

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v3_en_mobile/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v3_en_mobile/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v3_en_mobile/best_accuracy/"
    
    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v2_en_mobile_pp-OCRv2/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v2_en_mobile_pp-OCRv2/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v2_en_mobile_pp-OCRv2/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite_no_pretrain/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite_no_pretrain/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite_no_pretrain/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite_tc_res_mobile/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite_tc_res_mobile/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite_tc_res_mobile/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite_cnn/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite_cnn/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_lite_cnn/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_dnn/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_dnn/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_dnn/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_rnn/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_rnn/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_rnn/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_enhanced_ctc/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_enhanced_ctc/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_enhanced_ctc/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_distillation/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_distillation/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_distillation/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_distillation/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_distillation/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_distillation/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_gray/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_gray/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_gray/best_accuracy/"
    
    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresizeratio_gray/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresizeratio_gray/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresizeratio_gray/best_accuracy/"


    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_center_rmresize_ratio_gray/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_center_rmresize_ratio_gray/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_center_rmresize_ratio_gray/best_accuracy/"
    
    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320_1215_simple/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320_1215_simple/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320_1215_simple/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_simple/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_simple/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_simple/best_accuracy/"

    # # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_center_rmresize_ratio_gray_64_320_1215_simple/config.yml"
    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_center_rmresize_ratio_gray_64_320_1215_simple/best_accuracy"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_center_rmresize_ratio_gray_64_320_1215_simple/best_accuracy/"

    # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_all/config.yml"
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_all/best_accuracy"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_all/best_accuracy/"

    # # ocr_merge_test
    # # args.img_list = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/ocr_merge_test/ImageSets/Main/test.txt"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test/ocr_merge_test_result.csv')
    # args.img_list = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/data_crop_1024_1029/ImageSets/Main/trainval.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'trainval_1024_1029/ocr_merge_test_result.csv')
    # model_test(args)

    # # from_jpg_dir
    # # args.input_jpg_path = "/yuanhuan/data/image/ZD_anpr/test_video/ZD_DUBAI/jpg文件/特殊车牌_crop/豹子号"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test/data_synthesis_baozihao_result.csv')
    # # model_test(args, from_jpg_dir=True)

    ###############################
    # onnx
    ###############################

    # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile/config.yml"
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile/inference/onnx/model.onnx"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile/best_accuracy/"

    # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize/config.yml"
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize/inference/onnx/model.onnx"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize/best_accuracy/"

    # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray/config.yml"
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray/inference/onnx/model.onnx"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray/best_accuracy/"

    # args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_1209/config.yml"
    # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_1209/inference/onnx/model.onnx"
    # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_1209/best_accuracy/"

    args.config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_all/config.yml"
    args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_all/inference/onnx/model.onnx"
    args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_all/best_accuracy/"

    # ocr_merge_test
    # args.img_list = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/ocr_merge_test/ImageSets/Main/test.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_onnx/ocr_merge_test_result.csv')
    args.img_list = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/data_crop_1024_1029/ImageSets/Main/trainval.txt"
    args.output_csv_path = os.path.join(args.output_dir, 'trainval_1024_1029_onnx/ocr_merge_test_result.csv')
    model_test(args, bool_onnx=True)

    # # from_jpg_dir
    # args.input_jpg_path = "/yuanhuan/data/image/ZD_anpr/test_video/ZD_DUBAI/jpg文件/特殊车牌_crop/豹子号"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_onnx/data_synthesis_baozihao_result.csv')
    # model_test(args, from_jpg_dir=True, bool_onnx=True)


    # ###############################
    # # caffe
    # ###############################

    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize/inference/caffe/deploy.ng.caffemodel"
    # # args.prototxt_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize/inference/caffe/deploy.ng.prototxt"
    # # args.dict_path = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/type/zd_dict.txt"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize/best_accuracy/"

    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray/inference/caffe/model-sim-clip-rename.caffemodel"
    # # args.prototxt_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray/inference/caffe/model-sim-clip-rename.prototxt"
    # # args.dict_path = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/type/zd_dict.txt"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray/best_accuracy/"

    # # args.model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_1209/inference/caffe/model-clip-sim-rename.caffemodel"
    # # args.prototxt_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_1209/inference/caffe/model-clip-sim-rename.prototxt"
    # # args.dict_path = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/type/zd_dict.txt"
    # # args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_1209/best_accuracy/"

    # # ocr_merge_test
    # # args.img_list = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/ocr_merge_test/ImageSets/Main/test.txt"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/ocr_merge_test_result.csv')
    # args.img_list = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/data_crop_1024_1029/ImageSets/Main/trainval.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'trainval_1024_1029_caffe/ocr_merge_test_result.csv')
    # model_test(args, bool_caffe=True)

    # # # from_jpg_dir
    # # args.input_jpg_path = "/yuanhuan/data/image/ZD_anpr/test_video/ZD_DUBAI/jpg文件/特殊车牌_crop/豹子号"
    # # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_synthesis_baozihao_result.csv')
    # # model_test(args, from_jpg_dir=True, bool_caffe=True)
