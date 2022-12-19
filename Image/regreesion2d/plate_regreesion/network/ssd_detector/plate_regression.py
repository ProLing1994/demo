import os
import sys
from importlib import import_module

import numpy as np
import torch

from ssd_detector.car_regression import img_crop
import config.config as config


class PlateRegression(object):
    def __init__(self, pt_path, device='cpu'):
        # sys.path.append(f"/{os.path.join(*pt_path.split('/')[:-4])}")
        # config = import_module(".".join(pt_path.split("/")[-4:-1]) + ".config")
        cfg = config.Config()
        cfg = config.Config()
        self.cfg = cfg
        # self.cfg = Config()
        # cfg = self.cfg
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


if __name__ == '__main__':
    print("aa")
    #
    # cfg = Config()
    # pt_path = "ssd_detector/MobileNetSmallV1_Hunan_2019_11_28_20_45_19_38.pt"
    # mat_path = "../data/voc_bbox_over_crop/test2_crop.mat"
    # # mat_path = "../data/plate_voc/test_crop.mat"
    # test_set = DataSetAHD(mat_path, mean=cfg.img_mean, scale=cfg.img_scale, mode=cfg.color_mode)
    # # test_set = DataSetAHD(mat_path, mean=128, scale=256, mode="rgb")
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
    #                                           shuffle=False, num_workers=1)
    # dataiter = iter(test_loader)
    # if os.path.splitext(pt_path)[1] == '.pt':
    #     net = cfg.model(n_class=cfg.num_class, width_mult=cfg.width_mul, input_channel=cfg.input_channel,
    #                     last_channel=cfg.output_channel, interverted_residual_setting=cfg.interverted_residual_setting)
    #     # net = MobileNetV2(n_class=3)
    #     net.load_state_dict(torch.load(pt_path), strict=False)
    # else:
    #     net = torch.load(pt_path)
    # net.eval()
