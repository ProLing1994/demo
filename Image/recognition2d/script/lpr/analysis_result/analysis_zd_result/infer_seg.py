import argparse
import cv2
import importlib
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
from lpr.script.dataset.dataset_seg_zd.dataset_mask.gen_seg_mask import get_kind_num_id
from lpr.infer.lpr_seg import LPRSegCaffe, LPRSegPytorch, LPRSeg2HeadPytorch, LPRSegColorClassPytorch
from script.lpr.dataset.dataset_zd.dataset_mask.gen_ocr_img import mask_2_bbox


def city_mask_2_bbox(mask, dataset_dict):
    
    seg_bbox = {}

    for key in dataset_dict.class_seg_label_group_2_id_map.keys():

        if key != "kind" and key != "num" and key != "country" and key != "city" and key != "car_type":
            continue

        label_list = dataset_dict.class_seg_label_group_2_id_map[key]
        label_mask = np.array((0,0))

        for idy in range(len(label_list)): 
            analysis_label = label_list[idy]
            label_mask = (mask == dataset_dict.id_2_mask_id_dict[analysis_label])[:,:,0].astype(np.uint8)
            label_size = label_mask.sum()
        
            if label_size > 0.0:

                label_bbox = mask_2_bbox(label_mask)[0]
                seg_bbox[dataset_dict.id_2_mask_name_dict[analysis_label]] = [[label_bbox[0], label_bbox[1], label_bbox[2], label_bbox[3]]]

    return seg_bbox


def color_mask_2_bbox(mask, dataset_dict):
    
    seg_bbox = {}

    for key in dataset_dict.class_seg_label_group_2_id_map.keys():

        if key != "color":
            continue

        label_list = dataset_dict.class_seg_label_group_2_id_map[key]
        label_mask = np.array((0,0))

        for idy in range(len(label_list)): 
            analysis_label = label_list[idy]
            label_mask = (mask == dataset_dict.id_2_mask_id_dict[analysis_label])[:,:,0].astype(np.uint8)
            label_size = label_mask.sum()
        
            if label_size > 0.0:

                label_bbox = mask_2_bbox(label_mask)[0]
                seg_bbox[dataset_dict.id_2_mask_name_dict[analysis_label]] = [[label_bbox[0], label_bbox[1], label_bbox[2], label_bbox[3]]]

    return seg_bbox


def model_test_color_class(args):

    # city
    dataset_city_dict = importlib.import_module(args.seg_city_dict_name) 
    # color
    dataset_color_dict = importlib.import_module(args.seg_color_dict_name)

    # mkdir 
    create_folder(args.output_dir)
    create_folder(args.output_mask_city)
    create_folder(args.output_mask_color)
    create_folder(args.output_mask_img_city)
    create_folder(args.output_mask_img_color)
    create_folder(args.output_bbox_img)

    # init 
    if args.caffe_bool:
        raise NotImplementedError
    elif args.pytorch_bool:
        lpr_seg = LPRSegColorClassPytorch(args.seg_pth_path, args.seg_city_dict_name, args.seg_color_dict_name)

    # img list
    img_list = []
    city_mask_list = []
    color_mask_list = []

    with open(args.input_test_file_path) as f:
        for line in f:
            img_list.append(line.strip().split(".jpg ")[0] + '.jpg')
            city_mask_list.append(line.split(".jpg ")[1].split(".png")[0] + '.png')
            color_mask_list.append(line.split(".jpg ")[1].split(".png ")[1].strip())

    # results list 
    results_list = []

    for idx in tqdm(range(len(img_list))):
        # init 
        results_dict = {}

        img_path = img_list[idx]
        img_name = os.path.basename(img_path)
        tqdm.write(img_path)

        img = cv2.imread(img_path) 

        # info 
        image_width = img.shape[1]
        image_height = img.shape[0]
        
        # run
        preds_mask_1, preds_mask_2, seg_mask_1, seg_mask_2, seg_bbox, seg_info = lpr_seg.run(img)

        # mask_img
        # city
        mask_img = np.zeros(img.shape, dtype=img.dtype)
        for mask_id in dataset_city_dict.id_2_mask_color_dict.keys():
            mask_img[seg_mask_1 == mask_id] = dataset_city_dict.id_2_mask_color_dict[mask_id]
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.3, gamma=0.)
        output_mask_img_path = os.path.join(args.output_mask_img_city, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_img_path, mask_img)
        # color
        mask_img = np.zeros(img.shape, dtype=img.dtype)
        for mask_id in dataset_color_dict.id_2_mask_color_dict.keys():
            mask_img[seg_mask_2 == mask_id] = dataset_color_dict.id_2_mask_color_dict[mask_id]
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=1.0, gamma=0.)
        output_mask_img_path = os.path.join(args.output_mask_img_color, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_img_path, mask_img)
        
        # mask
        # city
        mask = np.zeros((64, 128, 3), dtype=preds_mask_1.dtype)
        for mask_id in dataset_city_dict.id_2_mask_id_dict.keys():
            mask[preds_mask_1 == mask_id] = dataset_city_dict.id_2_mask_id_dict[mask_id]
        output_mask_path = os.path.join(args.output_mask_city, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_path, mask)
        # color
        mask = np.zeros((64, 128, 3), dtype=preds_mask_2.dtype)
        for mask_id in dataset_color_dict.id_2_mask_id_dict.keys():
            if preds_mask_2.max() == mask_id:
                mask[:, :] = dataset_color_dict.id_2_mask_id_dict[mask_id]
        output_mask_path = os.path.join(args.output_mask_color, img_name.replace(".jpg", ".png"))
        cv2.imwrite(output_mask_path, mask)

        # bbox_img
        for key in seg_bbox.keys():
            bbox = seg_bbox[key][0]

            # city
            if key in dataset_city_dict.name_2_mask_color_dict:
                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=dataset_city_dict.name_2_mask_color_dict[key], thickness=2)
            # color
            elif key in dataset_color_dict.name_2_mask_color_dict:
                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=dataset_color_dict.name_2_mask_color_dict[key], thickness=2)
        output_bbox_img_path = os.path.join(args.output_bbox_img, img_name)
        cv2.imwrite(output_bbox_img_path, img)

        # result
        # pd 
        results_dict['file'] = img_list[idx]
        results_dict['width'] = image_width
        results_dict['height'] = image_height
        # city
        for label_idx in range(len(dataset_city_dict.class_seg_label_group)):
            results_dict[dataset_city_dict.class_seg_label_group[label_idx]] = 'none'
            results_dict[dataset_city_dict.class_seg_label_group[label_idx] + '_res'] = 'none'
        # color
        for label_idx in range(len(dataset_color_dict.class_seg_label_group)):
            results_dict[dataset_color_dict.class_seg_label_group[label_idx]] = 'none'
            results_dict[dataset_color_dict.class_seg_label_group[label_idx] + '_res'] = 'none'

        # mask
        city_mask = cv2.imread(city_mask_list[idx])
        color_mask = cv2.imread(color_mask_list[idx])

        city_bbox = city_mask_2_bbox(city_mask, dataset_city_dict)
        color_bbox = color_mask_2_bbox(color_mask, dataset_color_dict)

        ## 获得 kind & num & country & city & car_type & color 字段
        for classname in city_bbox.keys():
            
            # city
            for label_idx in range(len(dataset_city_dict.class_seg_label_group)):
                if classname in dataset_city_dict.class_seg_label_group_2_name_map[dataset_city_dict.class_seg_label_group[label_idx]]:
                    results_dict[dataset_city_dict.class_seg_label_group[label_idx]] = classname

        for classname in color_bbox.keys():
            # color
            for label_idx in range(len(dataset_color_dict.class_seg_label_group)):
                if classname in dataset_color_dict.class_seg_label_group_2_name_map[dataset_color_dict.class_seg_label_group[label_idx]]:
                    results_dict[dataset_color_dict.class_seg_label_group[label_idx]] = classname

        ## 获得模型预测结果字段
        # city
        for label_idx in range(len(dataset_city_dict.class_seg_label_group)):
            results_dict[dataset_city_dict.class_seg_label_group[label_idx] + '_res'] = seg_info[dataset_city_dict.class_seg_label_group[label_idx]]
        # color
        for label_idx in range(len(dataset_color_dict.class_seg_label_group)):
            results_dict[dataset_color_dict.class_seg_label_group[label_idx] + '_res'] = seg_info[dataset_color_dict.class_seg_label_group[label_idx]]
            
        results_list.append(results_dict)

    # out csv
    csv_data_pd = pd.DataFrame(results_list)
    csv_data_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()


    ###############################################
    # dataset_zd_dict_city & dataset_zd_dict_color
    # CityColorClassSeg
    ###############################################

    args.caffe_bool = False
    args.pytorch_bool = True

    # zd: seg_city_color_class_zd_20230703
    args.seg_caffe_prototxt = ""
    args.seg_caffe_model_path = "" 
    args.seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230703/LaneNetNova2Head.pth"
    args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_color_class_zd_20230703/"
    args.seg_city_dict_name = "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_city"
    args.seg_color_dict_name = "script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_color"

    # from dataset
    args.input_dir = "/yuanhuan/data/image/RM_ANPR/training/seg_zd_202307/"
    args.dataset_name = "ImageSetsLabelNoAug/city_color_label"
    args.mode = "test"
    args.input_test_file_path = os.path.join(args.input_dir, args.dataset_name, "ImageSets/Main/{}.txt".format(args.mode))
    
    args.output_csv_path = os.path.join(args.output_dir, '{}/{}/result.csv'.format(args.mode, args.dataset_name))
    args.output_mask_city = os.path.join(args.output_dir, '{}/{}/mask_city'.format(args.mode, args.dataset_name))
    args.output_mask_color = os.path.join(args.output_dir, '{}/{}/mask_color'.format(args.mode, args.dataset_name))
    args.output_mask_img_city = os.path.join(args.output_dir, '{}/{}/mask_img_city'.format(args.mode, args.dataset_name))
    args.output_mask_img_color = os.path.join(args.output_dir, '{}/{}/mask_img_color'.format(args.mode, args.dataset_name))
    args.output_bbox_img = os.path.join(args.output_dir, '{}/{}/bbox_img'.format(args.mode, args.dataset_name))

    model_test_color_class(args)