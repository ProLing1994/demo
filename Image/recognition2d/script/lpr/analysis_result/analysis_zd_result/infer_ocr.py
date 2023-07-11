import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys 
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from lpr.infer.lpr import LPRCaffe, LPRPytorch


def ocr_labels_init(args):

    character_str = []
    character_str.append(" ")
    with open(args.dict_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.decode('utf-8').strip("\n").strip("\r\n")
            character_str.append(line)
        character_str.append(" ")
    ocr_labels = list(character_str)
    return ocr_labels


def model_test(args):
    
    # mkdir 
    create_folder(os.path.dirname(args.output_csv_path))

    # ocr_labels_init
    ocr_labels = ocr_labels_init(args)
    
    # lpr
    if args.caffe_bool:
        lpr = LPRCaffe(args.lpr_caffe_prototxt, args.lpr_caffe_model_path, input_shape=(256, 64), ocr_labels=ocr_labels, prefix_beam_search_bool=args.lpr_prefix_beam_search_bool, gpu_bool=args.gpu_bool)
    elif args.pytorch_bool:
        lpr = LPRPytorch(args.lpr_pth_path, input_shape=(256, 64), ocr_labels=ocr_labels, prefix_beam_search_bool=args.lpr_prefix_beam_search_bool, gpu_bool=args.gpu_bool)

    # img list
    img_list = []

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

        img = cv2.imread(img_path)        

        # info 
        image_width = img.shape[1]
        image_height = img.shape[0]

        # ocr 
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ocr, ocr_score = lpr.run(gray_img)

        # pd 
        results_dict['file'] = img_path
        results_dict['width'] = image_width
        results_dict['height'] = image_height
        results_dict['label'] = img_name.replace('.jpg', '').split('_')[-1]

        results_dict['ocr'] = ocr
        results_dict['ocr_score'] = np.array(ocr_score).mean()
        results_dict['res'] = int( results_dict['label'] == results_dict['ocr'] )
        # results_dict['res'] = int( results_dict['label'][1:] == results_dict['ocr'][1:] )
        
        tqdm_write = '{} {} {}'.format( results_dict['label'], results_dict['ocr'], int( results_dict['label'] == results_dict['ocr'] ) )
        tqdm.write(tqdm_write)

        results_list.append(results_dict)

    # out csv
    csv_data_pd = pd.DataFrame(results_list)
    csv_data_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    args.caffe_bool = True
    args.pytorch_bool = False
    # cn: v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230530_cn
    args.lpr_caffe_prototxt = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230530_cn/inference/caffe/model.prototxt"
    args.lpr_caffe_model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230530_cn/inference/caffe/model.caffemodel"
    args.lpr_pth_path = ""
    args.dict_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230530_cn/inference/cn_dict.txt"
    args.output_dir = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20230530_cn/best_accuracy/"

    args.lpr_prefix_beam_search_bool = False
    args.gpu_bool = False

    # ocr_merge_test
    # args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_cn_202305/sichuan/ImageSets/Main/train.txt"
    # args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_sichuan_train_result.csv')
    args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_cn_202305/sichuan/ImageSets/Main/test.txt"
    args.output_csv_path = os.path.join(args.output_dir, 'test_caffe/data_sichuan_test_result.csv')
    model_test(args)