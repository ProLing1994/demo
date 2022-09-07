from re import I
import cv2
import numpy as np
import os
import random
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Stitching.utils.json_code import load_bbox_json
from Image.Stitching.utils.xml_code import load_bbox_xml
from Image.Stitching.utils.image_processing import *


class ImageSitchApi():
    """
    ImageSitchingApi
    """

    def __init__(self, bkg_path, bkg_roi):
    
        self.bkg_img = cv2.imread(bkg_path)
        self.bkg_roi = np.array(bkg_roi)

        # option
        self.option_init()


    def option_init(self):
        
        # annotation
        self.annotation_bbox_bool = True

        # params
        # 最大贴图数目
        self.max_sitch_num = 30
        
        # 贴图处理方法
        # self.sitch_pitch_processing_method_list = ['']
        # self.sitch_pitch_processing_method_list = ['otsu']
        self.sitch_pitch_processing_method_list = ['Gaussian&otsu']
        # self.sitch_pitch_processing_method_list = ['otsu', 'Gaussian&otsu']

        # 贴图融合方法
        self.sitch_img_method_list = ['cover']
        # self.sitch_img_method_list = ['laplace_pyramid']
    

    def run(self, img, annotation_path, annotation_type):

        if self.annotation_bbox_bool:
            img = self.run_bbox(img, annotation_path, annotation_type)

        return img


    def run_bbox(self, img, annotation_path, annotation_type):

        bbox_list = self.load_bbox(annotation_path, annotation_type)
        sitch_num = min(random.randint(1, self.max_sitch_num), len(bbox_list))
        sitch_bbox_list = random.sample(bbox_list, sitch_num)

        sitch_pitch_list = self.get_sitch_pitch(img, sitch_bbox_list)
        
        sitch_img, sitch_mask, stich_img_list = self.get_sitch_img(sitch_pitch_list)

        sitch_res_img = self.sitch_bkg_img(sitch_img, sitch_mask)

        return sitch_res_img, stich_img_list


    def load_bbox(self, annotation_path, annotation_type):

        if annotation_type == "json":
            bbox_list = load_bbox_json(annotation_path)
        elif annotation_type == "xml":
            bbox_list = load_bbox_xml(annotation_path)
        else:
            raise Exception
        return bbox_list
    

    def get_sitch_pitch(self, img, sitch_bbox_list):
        
        sitch_pitch_list = [] # [{'label': label, 'img': img, "mask": mask}]

        for idx in range(len(sitch_bbox_list)):
            sitch_bbox_idx = sitch_bbox_list[idx]

            sitch_pitch = img[sitch_bbox_idx['bbox'][1] : sitch_bbox_idx['bbox'][3], sitch_bbox_idx['bbox'][0] : sitch_bbox_idx['bbox'][2]]
            sitch_pitch, sitch_pitch_mask = self.sitch_pitch_processing(sitch_pitch)
            
            sitch_pitch_list.append({"label": sitch_bbox_idx['label'], "img":sitch_pitch, "mask":sitch_pitch_mask})

            output_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_res/"
            output_img_path = os.path.join(output_dir, "sitch_pitch_{}.jpg".format(idx))
            cv2.imwrite(output_img_path, sitch_pitch)

        return sitch_pitch_list

    
    def sitch_pitch_processing(self, img):
        
        sitch_pitch_processing_method = random.sample(self.sitch_pitch_processing_method_list, 1)[0]
        
        if sitch_pitch_processing_method == "":
            img = img
            mask = np.ones(img.shape)
        elif sitch_pitch_processing_method == "otsu":
            img, mask = otsu(img)
        elif sitch_pitch_processing_method == "Gaussian&otsu":
            img, mask = gaussian_otsu(img)
        
        return img, mask


    def get_pitch_roi(self, img, mask, pitch_img, pitch_mask):
        
        num = 0
        bool_find_pitch_roi = False

        while not bool_find_pitch_roi:
            x1 = random.randint(0, img.shape[1] - pitch_img.shape[1])
            y1 = random.randint(0, img.shape[0] - pitch_img.shape[0])
            x2 = x1 + pitch_img.shape[1]
            y2 = y1 + pitch_img.shape[0]

            bool_in_bkg_roi = cv2.pointPolygonTest(self.bkg_roi, (x1, y1), False) >= 0
            bool_not_in_mask = mask[y1:y2, x1:x2, :].sum() == 0
            if bool_not_in_mask and bool_in_bkg_roi:
                bool_find_pitch_roi = True
                mask[y1:y2, x1:x2] = pitch_mask
        
            if num >= 30:
                break

            num += 1

        return [x1, y1, x2, y2], bool_find_pitch_roi
        

    def get_sitch_img(self, sitch_pitch_list):

        sitch_img = np.zeros(self.bkg_img.shape, np.uint8)
        sitch_mask = sitch_img.copy()
        stich_img_list = [] # [{'label': label, 'bbox': [x1, y1, x2, y2]}]

        for idx in range(len(sitch_pitch_list)):

            sitch_pitch_idx = sitch_pitch_list[idx]

            pitch_roi, bool_find_pitch_roi = self.get_pitch_roi(sitch_img, sitch_mask, sitch_pitch_idx['img'], sitch_pitch_idx['mask'])

            if bool_find_pitch_roi:
                sitch_img[pitch_roi[1]:pitch_roi[3], pitch_roi[0]:pitch_roi[2]] = sitch_pitch_idx['img']
                stich_img_list.append({'label': sitch_pitch_idx['label'], 'bbox': [pitch_roi[0], pitch_roi[1], pitch_roi[2], pitch_roi[3]]})

        return sitch_img, sitch_mask, stich_img_list

    
    def sitch_bkg_img(self, sitch_img, sitch_mask):
        
        sitch_method = random.sample(self.sitch_img_method_list, 1)[0]

        if sitch_method == 'cover':
            bkg_img = self.bkg_img.copy()
            bkg_img = bkg_img * ( 1 - sitch_mask )
            sitch_res_img = cv2.addWeighted(src1=bkg_img, alpha=1.0, src2=sitch_img, beta=1.0, gamma=0.)
        elif sitch_method == 'laplace_pyramid':
            bkg_img = self.bkg_img.copy()
            sitch_res_img = sitch_laplace_pyramid(bkg_img, sitch_img, sitch_mask)
            
        
        return sitch_res_img
        
