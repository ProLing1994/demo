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
from lpr.infer.lpr import LPRCaffe, LPRPytorch, ocr_labels_zd


def model_test(args):
    
    # mkdir 
    create_folder(os.path.dirname(args.output_csv_path))

    # lpr
    if args.caffe_bool:
        lpr = LPRCaffe(args.lpr_caffe_prototxt, args.lpr_caffe_model_path, input_shape=(256, 64), ocr_labels=ocr_labels_zd, prefix_beam_search_bool=args.lpr_prefix_beam_search_bool)
    elif args.pytorch_bool:
        lpr = LPRPytorch(args.lpr_pth_path, input_shape=(256, 64), ocr_labels=ocr_labels_zd, prefix_beam_search_bool=args.lpr_prefix_beam_search_bool)

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
    
    args.caffe_bool = False
    args.pytorch_bool = True

    # zd: ocr_zd_mask_pad_20230703
    args.lpr_caffe_prototxt = ""
    args.lpr_caffe_model_path = ""
    args.lpr_pth_path = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_pad_20230703/crnn_best.pth"
    args.output_dir = "/yuanhuan/model/image/lpr/zd/ocr_zd_mask_pad_20230703/"

    args.lpr_prefix_beam_search_bool = False
    # args.lpr_prefix_beam_search_bool = True

    # ocr_merge_test
    args.img_list = "/yuanhuan/data/image/RM_ANPR/training/plate_zd_mask_202307/ImageSetsOcrLabelNoAug/ImageSets/Main/test.txt"
    args.output_csv_path = os.path.join(args.output_dir, 'test/ocr_merge_test_result.csv')
    model_test(args)