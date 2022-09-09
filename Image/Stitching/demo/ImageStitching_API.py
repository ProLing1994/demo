from array import array
from re import I
import cv2
import numpy as np
import os
import random
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Stitching.utils.json_code import load_bbox_json, load_mask_json
from Image.Stitching.utils.xml_code import load_bbox_xml
from Image.Stitching.utils.image_processing import *


class ImageSitchApi():
    """
    ImageSitchingApi
    """

    def __init__(self, bkg_path_list, bkg_roi_list, img_path_list, annotation_path_list, bbox_mask_type, json_xml_type):
        """
        初始化
        """
     
        # option
        self.option_init()

        # load bkg
        self.bkg_img_list = []
        self.bkg_roi_list = []
        self.load_bkg(bkg_path_list, bkg_roi_list)

        # load img
        self.sitch_pitch_list = []
        self.load_img(img_path_list, annotation_path_list, bbox_mask_type, json_xml_type)


    def option_init(self):
        
        # params
        # 最大贴图数目
        self.max_sitch_num = 50
        
        # 贴图增强方法
        self.sitch_pitch_aug_rotate_angle_list = range(-20, 20)
        self.sitch_pitch_aug_rotate_mirror_list = [0, 1]
        self.sitch_pitch_aug_rotate_scale_list = list(np.arange(0.8, 1.2, 0.01))

        # 贴图处理方法
        # self.sitch_pitch_processing_method_list = ['']
        # self.sitch_pitch_processing_method_list = ['otsu']
        self.sitch_pitch_processing_method_list = ['Gaussian&otsu']
        # self.sitch_pitch_processing_method_list = ['otsu', 'Gaussian&otsu']

        # 忽略贴图不太好的小块
        self.sitch_pitch_processing_ignore_bool = True

        # 贴图融合方法
        self.sitch_img_method_list = ['cover']
        # self.sitch_img_method_list = ['laplace_pyramid']
    

    def load_bkg(self, bkg_path_list, bkg_roi_list):
        
        assert len(bkg_path_list) == len(bkg_roi_list)

        for idx in range(len(bkg_path_list)):

            self.bkg_img_list.append(cv2.imread(bkg_path_list[idx]))
            self.bkg_roi_list.append(np.array(bkg_roi_list[idx]))


    def load_img(self, img_path_list, annotation_path_list, bbox_mask_type, json_xml_type):

        assert len(img_path_list) == len(annotation_path_list)
        assert bbox_mask_type in ['bbox', 'mask']
        
        for idx in range(len(img_path_list)):

            img = cv2.imread(img_path_list[idx])

            if bbox_mask_type == "bbox":
                bbox_list = self.load_bbox(annotation_path_list[idx], json_xml_type)
                self.sitch_pitch_list.extend(self.get_sitch_pitch_from_bbox(img, bbox_list))
            elif bbox_mask_type == "mask":
                mask_list = self.load_mask(annotation_path_list[idx], json_xml_type)
                self.sitch_pitch_list.extend(self.get_sitch_pitch_from_mask(img, mask_list))


    def load_bbox(self, annotation_path, json_xml_type):
    
        if json_xml_type == "json":
            bbox_list = load_bbox_json(annotation_path)
        elif json_xml_type == "xml":
            bbox_list = load_bbox_xml(annotation_path)
        else:
            raise Exception
        return bbox_list
    

    def load_mask(self, annotation_path, json_xml_type):

        if json_xml_type == "json":
            mask_list = load_mask_json(annotation_path)
        else:
            raise Exception

        return mask_list


    def add_bkg(self, bkg_path_list, bkg_roi_list):
        """
        添加背景图片
        """
        self.load_bkg(bkg_path_list, bkg_roi_list)


    def add_img(self, img_path_list, annotation_path_list, bbox_mask_type, json_xml_type):
        """
        添加拉结图片
        """
        self.load_img(img_path_list, annotation_path_list, bbox_mask_type, json_xml_type)


    def run(self):

        # 获得贴图小块    
        sitch_num = min(random.randint(1, self.max_sitch_num), len(self.sitch_pitch_list))
        sitch_pitch_list = random.sample(self.sitch_pitch_list, sitch_num)

        # 获得背景图像
        self.get_bkg_img()
        
        # 获得贴图图像
        sitch_img, sitch_mask, stich_label_list = self.get_sitch_img(sitch_pitch_list)

        # 获得最终结果，融合背景图像
        sitch_res_img = self.sitch_bkg_img(sitch_img, sitch_mask)

        return sitch_res_img, stich_label_list


    def get_sitch_pitch_from_bbox(self, img, bbox_list):
        """
        img: np.narray
        bbox_list: list [{'label': label, 'bbox': [x1, y1, x2, y2]}]

        return:
        sitch_pitch_list: list [{'label': label, 'img': img, "mask": mask, "corner":cornermask}]
        """
        sitch_pitch_list = [] # [{'label': label, 'img': img, "mask": mask, "corner":cornermask}]

        for idx in range(len(bbox_list)):
            bbox_idx = bbox_list[idx]

            # # 往外扩张
            # height, width, _ = img.shape
            # bbox_idx['bbox'][0] = max(0, bbox_idx['bbox'][0] - 20)
            # bbox_idx['bbox'][1] = max(0, bbox_idx['bbox'][1] - 20)
            # bbox_idx['bbox'][2] = min(width, bbox_idx['bbox'][2] + 20)
            # bbox_idx['bbox'][3] = min(height, bbox_idx['bbox'][3] + 20)

            # 贴图小块增强
            # 旋转 + 镜像 + 缩放
            rotate_img, rotate_bbox = sitch_pitch_aug_rotate_bbox(img, bbox_idx['bbox'], self.sitch_pitch_aug_rotate_angle_list)
            mirror_img, mirror_bbox = sitch_pitch_aug_mirror_bbox(rotate_img, rotate_bbox, self.sitch_pitch_aug_rotate_mirror_list)
            scaler_img, scale_bbox = sitch_pitch_aug_scale_bbox(mirror_img, mirror_bbox, self.sitch_pitch_aug_rotate_scale_list)

            # 前景提取：获得 贴图 和 mask
            # sitch_pitch = rotate_img[rotate_bbox[1] : rotate_bbox[3], rotate_bbox[0] : rotate_bbox[2]]
            # sitch_pitch = mirror_img[mirror_bbox[1] : mirror_bbox[3], mirror_bbox[0] : mirror_bbox[2]]
            sitch_pitch = scaler_img[scale_bbox[1] : scale_bbox[3], scale_bbox[0] : scale_bbox[2]]
            sitch_pitch, sitch_pitch_mask = sitch_pitch_foreground_extract(sitch_pitch, self.sitch_pitch_processing_method_list)

            # corner
            sitch_pitch_corner, hierarchy = cv2.findContours(sitch_pitch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sitch_pitch_corner = np.squeeze(np.concatenate(sitch_pitch_corner, axis=0))

            # 剔除不合格的小块
            if self.sitch_pitch_processing_ignore_bool:
                if sitch_pitch_corner.shape[0] <= 15:
                    continue
            
            sitch_pitch_list.append({"label": bbox_idx['label'], "img":sitch_pitch, "mask":sitch_pitch_mask, "corner":sitch_pitch_corner})

            output_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_res/"
            output_img_path = os.path.join(output_dir, "sitch_pitch_{}.jpg".format(idx))
            cv2.imwrite(output_img_path, sitch_pitch)

            # cv2.circle(scaler_img, (scale_bbox[0], scale_bbox[1]), 5, (0, 0, 255), 5)
            # cv2.circle(scaler_img, (scale_bbox[2], scale_bbox[3]), 5, (0, 0, 255), 5)
            # cv2.imwrite(output_img_path, scaler_img)

        return sitch_pitch_list


    def get_sitch_pitch_from_mask(self, img, mask_list):
        """
        img: np.narray
        mask_list: list [{'label': label, 'corner': [points]}]

        return:
        sitch_pitch_list: list [{'label': label, 'img': img, "mask": mask, "corner":cornermask}]
        """
        sitch_pitch_list = [] # [{'label': label, 'img': img, "mask": mask, "corner":cornermask}]

        for idx in range(len(mask_list)):

            mask_idx = mask_list[idx]

            # corner
            sitch_pitch_corner = np.array(mask_idx['corner'])

            # # 贴图小块增强
            # # 旋转 + 镜像 + 缩放
            rotate_img, rotate_corner = sitch_pitch_aug_rotate_mask(img, sitch_pitch_corner, self.sitch_pitch_aug_rotate_angle_list)
            mirror_img, mirror_corner = sitch_pitch_aug_mirror_mask(rotate_img, rotate_corner, self.sitch_pitch_aug_rotate_mirror_list)
            scaler_img, scale_corner = sitch_pitch_aug_scale_mask(mirror_img, mirror_corner, self.sitch_pitch_aug_rotate_scale_list)

            # bbox 
            sitch_pitch_bbox = [min(scale_corner[:, 0]), min(scale_corner[:, 1]),  max(scale_corner[:, 0]), max(scale_corner[:, 1])]

            # mask
            scale_corner = [scale_corner.reshape(-1, 1, 2)]
            rotate_mask = np.zeros((scaler_img.shape[0], scaler_img.shape[1]), dtype=img.dtype)
            rotate_mask = cv2.drawContours(rotate_mask, scale_corner, -1, 1, cv2.FILLED)
            sitch_pitch_mask = rotate_mask[sitch_pitch_bbox[1] : sitch_pitch_bbox[3], sitch_pitch_bbox[0] : sitch_pitch_bbox[2]]
            sitch_pitch_mask = np.expand_dims(sitch_pitch_mask, axis=2)

            # 贴图
            sitch_pitch = scaler_img[sitch_pitch_bbox[1] : sitch_pitch_bbox[3], sitch_pitch_bbox[0] : sitch_pitch_bbox[2]]
            sitch_pitch = sitch_pitch * sitch_pitch_mask
           
            sitch_pitch_list.append({"label": mask_idx['label'], "img":sitch_pitch, "mask":sitch_pitch_mask, "corner":sitch_pitch_corner})

            output_dir = "/mnt/huanyuan2/data/image/RM_DSLJ_detection/test_res/"
            output_img_path = os.path.join(output_dir, "sitch_pitch_{}.jpg".format(idx))
            cv2.imwrite(output_img_path, sitch_pitch)

        return sitch_pitch_list


    def get_bkg_img(self):
        
        bkg_id = random.randint(0, len(self.bkg_img_list) - 1)
        
        self.bkg_img = self.bkg_img_list[bkg_id]
        self.bkg_roi = self.bkg_roi_list[bkg_id]

        
    def get_sitch_img(self, sitch_pitch_list):

        sitch_img = np.zeros(self.bkg_img.shape, np.uint8)
        sitch_mask = sitch_img.copy()
        stich_label_list = [] # [{'label': label, 'bbox': [x1, y1, x2, y2], 'corner': [corner]}]

        for idx in range(len(sitch_pitch_list)):

            sitch_pitch_idx = sitch_pitch_list[idx]

            pitch_roi, pitch_corner, bool_find_pitch_roi = self.get_pitch_roi(sitch_img, sitch_mask, sitch_pitch_idx['img'], sitch_pitch_idx['mask'])

            if bool_find_pitch_roi:
                sitch_img[pitch_roi[1]:pitch_roi[3], pitch_roi[0]:pitch_roi[2]] = sitch_pitch_idx['img']
                stich_label_list.append({'label': sitch_pitch_idx['label'], 'bbox': [pitch_roi[0], pitch_roi[1], pitch_roi[2], pitch_roi[3]], 'corner': pitch_corner})

        return sitch_img, sitch_mask, stich_label_list


    def get_pitch_roi(self, img, mask, pitch_img, pitch_mask):
        
        num = 0
        bool_find_pitch_roi = False

        while not bool_find_pitch_roi:
            # bbox
            x1 = random.randint(0, img.shape[1] - pitch_img.shape[1])
            y1 = random.randint(0, img.shape[0] - pitch_img.shape[0])
            x2 = x1 + pitch_img.shape[1]
            y2 = y1 + pitch_img.shape[0]

            # corner
            corner, hierarchy = cv2.findContours(pitch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            corner = np.squeeze(np.concatenate(corner, axis=0))
            corner[:, 0] = corner[:, 0] + x1
            corner[:, 1] = corner[:, 1] + y1

            bool_in_bkg_roi = cv2.pointPolygonTest(self.bkg_roi, (x1, y1), False) >= 0
            bool_not_in_mask = mask[y1:y2, x1:x2, :].sum() == 0
            if bool_not_in_mask and bool_in_bkg_roi:
                bool_find_pitch_roi = True
                mask[y1:y2, x1:x2] = pitch_mask
        
            if num >= 30:
                break

            num += 1

        return [x1, y1, x2, y2], corner, bool_find_pitch_roi


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


def bbox_to_corner(img, bbox_list):
    """
    img: np.narray
    bbox_list: list [{'label': label, 'bbox': [x1, y1, x2, y2]}]

    return:
    sitch_pitch_list: list [{'label': label, 'img': img, "mask": mask, "corner":cornermask}]
    """
    output_list = [] # [{'label': label, 'bbox': [x1, y1, x2, y2], "corner":cornermask}]

    for idx in range(len(bbox_list)):
        bbox_idx = bbox_list[idx]

        # 前景提取：获得 贴图 和 mask
        sitch_pitch = img[bbox_idx['bbox'][1] : bbox_idx['bbox'][3], bbox_idx['bbox'][0] : bbox_idx['bbox'][2]]
        sitch_pitch, sitch_pitch_mask = sitch_pitch_foreground_extract(sitch_pitch, ['Gaussian&otsu'])

        # corner
        sitch_pitch_corner, hierarchy = cv2.findContours(sitch_pitch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sitch_pitch_corner = np.squeeze(np.concatenate(sitch_pitch_corner, axis=0))
        sitch_pitch_corner[:, 0] = sitch_pitch_corner[:, 0] + bbox_idx['bbox'][0]
        sitch_pitch_corner[:, 1] = sitch_pitch_corner[:, 1] + bbox_idx['bbox'][1]

        # 剔除不合格的小块，这里要酌情考虑，要不要一一对应
        if sitch_pitch_corner.shape[0] <= 15:
            continue
        
        output_list.append({"label": bbox_idx['label'], "bbox":bbox_idx['bbox'], "corner":sitch_pitch_corner})

    return output_list