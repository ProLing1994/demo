import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
# sys.path.insert(0, '/yuanhuan/code/demo')
from Image.recognition2d.license_plate_recognition.infer.lpr import LPR
from Image.Basic.utils.folder_tools import *


def inference_images(args):
    # lpr init
    lpr = LPR(args.lpr_caffe_prototxt, args.lpr_caffe_model_path, args.lpr_prefix_beam_search_bool)

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

        # ocr 
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ocr, ocr_score = lpr.run(gray_img)

        # pd 
        results_dict['file'] = img_list[idx]
        results_dict['label'] = str(img_list[idx]).split('@')[1].split('-')[0]
        # results_dict['label'] = str(img_list[idx]).split('_')[0]
        results_dict['ocr'] = ocr
        results_dict['ocr_score'] = ocr_score
        results_dict['res'] = int( results_dict['label'] == results_dict['ocr'] )
        # results_dict['res'] = int( results_dict['label'][1:] == results_dict['ocr'][1:] )
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
    
    # # china: lpr_lxn
    # args.lpr_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr_lxn/china_softmax.prototxt"
    # args.lpr_caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr_lxn/china.caffemodel"
    
    # china: lpr_zg
    args.lpr_caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr_zg/china/double/china_double_softmax.prototxt"
    args.lpr_caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr_zg/china/double/china_double.caffemodel"

    args.lpr_prefix_beam_search_bool = False
    # args.lpr_prefix_beam_search_bool = True
       
    # test
    args.img_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/test/"
    args.output_csv_path = "/mnt/huanyuan2/data/image/LicensePlateRecognition/test/test.csv"
    inference_images(args)


if __name__ == '__main__':
    main()