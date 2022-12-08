import cv2
from importlib import import_module
import numpy as np
import os
import random
import torch


def img_crop(img, annotation, rand_flag=True, rate_h=0.1, rate_w=0.1, bias_rate_h=0.1, bias_rate_w=0.2, crop_size_h=160,
             crop_size_w=224, inference_flag=False):
    pass
    # annotation:[xmin,ymin,width,height] (original number)
    # img: HWC image
    # RandFlag: do random crop or not
    # RateH, RateW : max enlarge rate (random)
    # CropSizeH, CropSizeW: Resize output image to a fix size

    H, W, _ = img.shape
    x = int(annotation[0])
    y = int(annotation[1])
    w = int(annotation[2])
    h = int(annotation[3])
    # print(H,W,x,y,w,h)
    if rand_flag:
        ran_w = 1 + (random.random() * rate_w + bias_rate_w)
        ran_h = 1 + (random.random() * rate_h + bias_rate_h)
        new_w = round(w * ran_w)
        new_h = round(h * ran_h)
        xx = random.randint(max(0, x - round(new_w - w) + round(w * bias_rate_w / 2)),
                            max(x - round(w * bias_rate_w / 2),
                                max(0, x - round(new_w - w) + round(w * bias_rate_w / 2))))
        yy = random.randint(max(0, y - round(new_h - h) + round(h * bias_rate_h / 2)),
                            max(y - round(h * bias_rate_h / 2),
                                max(0, y - round(new_h - h) + round(h * bias_rate_h / 2))))
    else:
        ran_w = 1 + bias_rate_w
        ran_h = 1 + bias_rate_h
        new_w = max(round(w * ran_w), 10)
        new_h = max(round(h * ran_h), 10)
        xx = max(round(x - (new_w - w) * 0.5), 0)
        yy = max(round(y - (new_h - h) * 0.5), 0)
    if yy + new_h > H:
        yy = H - new_h
    if xx + new_w > W:
        xx = W - new_w
    img_c = img[yy:yy + new_h, xx:xx + new_w, :]
    img_re = cv2.resize(img_c, (crop_size_w, crop_size_h))
    annotation_re = np.array(
        [float((x - xx) / (new_w / float(crop_size_w))), float((y - yy) / (new_h / float(crop_size_h))),
         w / (new_w / float(crop_size_w)), h / (new_h / float(crop_size_h))])
    if inference_flag:
        return img_re, np.array([xx, yy, new_w, new_h])
    return img_re, annotation_re


class PlateRegression(object):
    def __init__(self, pt_path, config_path, device='cpu'):
        dirname = os.path.dirname(config_path)
        basename = os.path.basename(config_path)
        modulename, _ = os.path.splitext(basename)

        os.sys.path.insert(0, dirname)
        config = import_module(modulename)
        os.sys.path.pop(0)

        cfg = config.Config()
        self.cfg = cfg
        self.net = cfg.model(n_class=cfg.num_class, width_mult=cfg.width_mul, input_channel=cfg.input_channel,
                             last_channel=cfg.output_channel,
                             interverted_residual_setting=cfg.interverted_residual_setting)
        self.net.load_state_dict(torch.load(pt_path), strict=False)
        self.net.eval()
        self.net.to(device)
        self.device = device
        self.img_height = cfg.img_height if hasattr(cfg, "img_height") else 160
        self.img_width = cfg.img_width if hasattr(cfg, "img_width") else 224

    def detect(self, img, det_boxes, plate_only=True):
        out_car_boxes = []
        out_plate_boxes = []
        for box in det_boxes:
            out = self.regression(img, box)
            out_car_boxes.append(out[0])
            out_plate_boxes.append(out[1])
        if plate_only:
            return out_plate_boxes
        else:
            return out_car_boxes, out_plate_boxes

    def regression(self, img, det_box):
        bbox_num = 1
        img_tensor = torch.zeros([bbox_num, 3, self.img_height, self.img_width])
        offset_array = np.zeros([bbox_num, 4])
        h, w, _ = img.shape
        left, top, right, bottom = det_box
        img_cup, anno = img_crop(img, [left, top, (right - left), (bottom - top)], rand_flag=False,
                                 bias_rate_h=0.2, bias_rate_w=0.2, crop_size_h=self.img_height,
                                 crop_size_w=self.img_width, inference_flag=True)
        img_tensor[0, :, :, :] = torch.from_numpy(
            np.reshape(
                np.transpose((img_cup.astype(np.float32) - self.cfg.img_mean) / self.cfg.img_scale, (2, 0, 1)),
                (1, 3, self.img_height, self.img_width)))
        offset_array[0, :] = anno
        pred_net = self.net(img_tensor.to(self.device))
        pred_net = pred_net.cpu().detach().numpy()
        xmin = int(round((pred_net[0, 0] / 2) * int(offset_array[0, 2]) + offset_array[0, 0]))
        xmax = int(round((pred_net[0, 1] / 2 + 0.5) * int(offset_array[0, 2]) + offset_array[0, 0]))
        ymax = int(round((pred_net[0, 2] / 2 + 0.5) * int(offset_array[0, 3]) + offset_array[0, 1]))
        xmin = max(xmin, 0)
        xmax = min(xmax, w)
        ymax = min(ymax, h)

        # predict[5], predict[6] = predict[5] / 5 * W, predict[6] / 5 * H
        # predict[3], predict[4] = predict[3] * W - predict[5] / 2, predict[4] * H - predict[6] / 2
        plate_w = max(int(pred_net[0, 5] / 5 * offset_array[0, 2] + 0.5), 0)
        plate_h = max(int(pred_net[0, 6] / 5 * offset_array[0, 3] + 0.5), 0)
        plate_l = max(int(pred_net[0, 3] * offset_array[0, 2] - plate_w / 2 + 0.5 + offset_array[0, 0]), 0)
        plate_t = max(int(pred_net[0, 4] * offset_array[0, 3] - plate_h / 2 + 0.5 + offset_array[0, 1]), 0)
        plate_r = plate_l + plate_w
        plate_b = plate_t + plate_h
        plate_r = min(plate_r, w)
        plate_b = min(plate_b, h)

        return [xmin, max(0, ymax - int((xmax - xmin) * 0.3)), xmax, ymax], [plate_l, plate_t, plate_r, plate_b]