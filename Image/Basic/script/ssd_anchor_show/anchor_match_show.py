import argparse
import numpy as np
import cv2
import os
import sys
import torch
from tqdm import tqdm

sys.path.insert(0, "/home/huanyuan/code/demo/Image/detection2d/ssd_rfb_crossdatatraining")
from data.voc0712 import VOCDetection
from data import *
from data.config import VOC_300
from layers.functions import PriorBox
from utils.box_utils import match, match_ATSS


def anchor_show(args):
    # init
    cfg = VOC_300
    threshold = 0.5
    variance = [0.1, 0.2]

    # mkdir 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # dataset
    dataset = VOCDetection(VOCroot, args.data_sets, preproc_show(300), AnnotationTransform())

    # priorbox
    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()

    num_priors = (priors.size(0))

    for index in tqdm(range(len(dataset))):
        # image & targets
        image, targets, _ = dataset[index]
        image = image.cpu().numpy().transpose(1, 2, 0)

        image_ori = dataset.pull_image(index)
        h, w , _ = image_ori.shape
        
        image = cv2.resize(image, (w, h))
        num = 1

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = torch.from_numpy(targets[:, :-1].astype(np.float32))
            labels = torch.from_numpy(targets[:, -1])
            defaults = priors.data

            if args.anchor_match == 'ATSS':
                match_ATSS(threshold, truths, defaults, variance, labels, loc_t, conf_t, idx, -1)
            else:
                match(threshold, truths, defaults, variance, labels, loc_t, conf_t, idx, -1)

        pos = conf_t > 0
        pos_idx = pos.squeeze(0).unsqueeze(1).expand_as(priors)
        anchor_t = priors[pos_idx].view(-1, 4)

        anchor_t[:,0::2] *= w
        anchor_t[:,1::2] *= h

        for i in range(anchor_t.shape[0]):
            xmin = int(anchor_t[i,0] - anchor_t[i,2]/2)
            ymin = int(anchor_t[i,1] - anchor_t[i,3]/2)
            xmax = int(anchor_t[i,0] + anchor_t[i,2]/2)
            ymax = int(anchor_t[i,1] + anchor_t[i,3]/2)
            p1 = (xmin, ymin)
            p2 = (xmax, ymax)
            cv2.rectangle(image, p1, p2, (0, 255, 0), 2)

        targets[:,0::2] *= w
        targets[:,1::2] *= h

        for i in range(targets.shape[0]):
            xmin = int(targets[i,0])
            ymin = int(targets[i,1])
            xmax = int(targets[i,2])
            ymax = int(targets[i,3])
            p1 = (xmin, ymin)
            p2 = (xmax, ymax)
            cv2.rectangle(image, p1, p2, (255, 0, 0),2)

        cv2.imwrite(os.path.join(args.output_dir, str(index) + ".jpg"), image)


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data_sets = [('LicensePlate/China', 'test', 0),]
    args.output_dir = "/mnt/huanyuan2/data/image/LicensePlate/China/anchor_match_show"
    args.anchor_match = 'ATSS'
    # args.anchor_match = 'SSD'
    anchor_show(args)


if __name__ == '__main__':
    main()