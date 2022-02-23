from src.util import post_processing

import torch
import numpy as np
from importlib import import_module
import cv2


class YoloDetector(object):
    def __init__(self, pt_path, device):
        # pt_path = "torch_model/YoloSmallLong_2020_01_03_16_06_31/model/14.pt"
        config = import_module(".".join(pt_path.split("/")[:-1]) + ".config")
        self.cfg = config.Config()
        self.device = device
        if isinstance(self.cfg.anchors_mask[0], tuple):
            num_anchors = len(self.cfg.anchors_mask[0])
        else:
            num_anchors = len(self.cfg.anchors_mask)
        self.net = self.cfg.model(num_classes=self.cfg.num_classes, num_anchors=num_anchors,
                                  backbone_block1_setting=self.cfg.backbone_block1_setting,
                                  backbone_block2_setting=self.cfg.backbone_block2_setting,
                                  yolo_head_setting=self.cfg.yolo_head_setting)
        self.net.load_state_dict(torch.load(pt_path), strict=False)
        self.net.to(device)
        self.net.eval()
        self.width_scale = 1
        self.height_scale = 1

    def detect(self, img, det_boxes, thresh=0.2, return_only_one=True):
        out_plate = []
        for box in det_boxes:
            out_plate.extend(self._detect_single_box(img, box, thresh, return_only_one))
        return out_plate

    def _detect_single_box(self, img, det_box, thresh=0.2, return_only_one=True):
        img_tensor = self._pre_processing(img.copy(), det_box)
        img_tensor = img_tensor.unsqueeze(dim=0).to(self.device)
        predict = self.net(img_tensor)
        # print(predict)
        nms_bbox_array = post_processing(None, predict, self.cfg.anchors, self.cfg.reduction,
                                         confidence_threshold=thresh, cfg=self.cfg, scale_h=self.height_scale,
                                         scale_w=self.width_scale)
        # plate_box = nms_bbox_array[0][:5]
        # plate_box[2:4] = [plate_box[2] + plate_box[0], plate_box[3] + plate_box[1]]
        # plate_box[:4] = [plate_box[i] + det_box[i % 2] for i in range(4)]
        out_box = []

        if return_only_one and len(nms_bbox_array) > 0:
            nms_bbox_array = [nms_bbox_array[0]]
        for box in nms_bbox_array:
            plate_box = box[:5]
            plate_box[2:4] = [plate_box[2] + plate_box[0], plate_box[3] + plate_box[1]]
            plate_box[:4] = [plate_box[i] + det_box[i % 2] for i in range(4)]
            out_box.append(plate_box)
        return out_box

    def _pre_processing(self, img, det_box):
        l, t, r, b = det_box[:4]
        img_crop = img[t:b, l:r, :]
        if self.cfg.color_mode == 'rgb':
            img_crop = img_crop[:, :, ::-1]
        img_h, img_w, _ = img_crop.shape
        self.width_scale = img_w / self.cfg.img_width
        self.height_scale = img_h / self.cfg.img_height
        if img_h != self.cfg.img_height or img_w != self.cfg.img_width:
            img_crop = cv2.resize(img_crop, (self.cfg.img_width, self.cfg.img_height))
        img_crop = (img_crop - self.cfg.img_mean) / self.cfg.img_scale
        return torch.from_numpy(np.transpose(img_crop.astype(np.float32), (2, 0, 1)))
