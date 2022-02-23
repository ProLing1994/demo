import torch
import numpy as np
import cv2
import random
from importlib import import_module


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


class CarRegression(object):
    def __init__(self, pt_path, device='cpu'):
        config = import_module(".".join(pt_path.split("/")[:-1]) + ".config")
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

    def regression(self, img, det_boxes):
        bbox_num = 1
        img_tensor = torch.zeros([bbox_num, 3, self.img_height, self.img_width])
        offset_array = np.zeros([bbox_num, 4])
        h, w, _ = img.shape
        left, top, right, bottom = det_boxes
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
        return [xmin, max(0, ymax - int((xmax - xmin) * 0.3)), xmax, ymax]


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
