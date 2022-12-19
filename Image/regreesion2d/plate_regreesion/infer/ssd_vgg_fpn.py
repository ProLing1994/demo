import cv2
import numpy as np
import os
import sys
import torch

# sys.path.insert(0, '/yuanhuan/code/demo/Image')
sys.path.insert(0, '/home/huanyuan/code/demo/Image')
from regreesion2d.plate_regreesion.network.ssd_detector.model import build_net
from regreesion2d.plate_regreesion.network.ssd_detector.utils.layer_function import Detect, PriorBox
from regreesion2d.plate_regreesion.network.ssd_detector.utils.nms_wrapper import nms
from regreesion2d.plate_regreesion.network.ssd_detector.utils.timer import Timer


VOC_CLASSES = ('__background__',  # always index 0
               'car', 'person')

VOC_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],

    'min_dim': 300,

    'steps': [8, 16, 32, 64, 100, 300],

    'min_sizes': [30, 60, 111, 162, 213, 264],

    'max_sizes': [60, 111, 162, 213, 264, 315],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],

    'variance': [0.1, 0.2],

    'clip': True,
}


class BaseTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, resize, rgb_means, swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img):
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[0]
        img = cv2.resize(np.array(img), (self.resize,
                                         self.resize), interpolation=interp_method).astype(np.float32)
        img -= self.means
        img = img.transpose(self.swap)
        return torch.from_numpy(img)


class SSDDetector(object):
    def __init__(self, num_classes=3, device="cpu", weight_path="ssd_detector/SSD_VGG_FPN_VOC_epoches_165.pth"):
        self.num_class = num_classes
        self.device = device
        cfg = VOC_300
        priorbox = PriorBox(cfg)
        with torch.no_grad():
            priors = priorbox.forward()
            self.priors = priors.to(device)
        self.img_dim = 300
        self.net = build_net('test', self.img_dim, self.num_class)
        self.net.load_state_dict(torch.load(weight_path))
        self.net.eval()
        self.net.to(device)
        self.detector = Detect(self.num_class, 0, cfg)
        self.top_k = 200
        self.rgb_means = (104, 117, 123)

    def detect(self, img, thresh=0.2, with_score=False):
        all_boxes = self._detect(img, self.net, self.detector, self.device,
                                 BaseTransform(self.net.size, self.rgb_means, (2, 0, 1)), self.top_k, thresh=thresh,
                                 num_classes=self.num_class)
        k = 1
        h, w, _ = img.shape
        out_dict = {}
        for i in range(1, len(all_boxes)):
            box_locations = []
            if len(all_boxes[k]) > 0:
                box_scores = all_boxes[k][:, 4]
                box_locations = all_boxes[k][:, 0:4]
                box_locations = [[int(b + 0.5) for b in box] for box in box_locations]
                bbox_out = []
                for box, score in zip(box_locations, box_scores):
                    l, t, r, b = box
                    l = max(l, 0)
                    t = max(t, 0)
                    r = min(w - 1, r)
                    b = min(h - 1, b)
                    if with_score:
                        bbox_out.append([l, t, r, b, score])
                    else:
                        bbox_out.append([l, t, r, b])
                box_locations = bbox_out
            out_dict[VOC_CLASSES[i]] = box_locations
        # else:
        #     box_scores = None
        #     box_locations = None

        return out_dict

    def _detect(self, img, net, detector, device, transform, max_per_image=300, thresh=0.5, num_classes=3):
        all_boxes = [[] for _ in range(num_classes)]

        _t = {'im_detect': Timer(), 'misc': Timer()}

        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            x = x.to(device)
            scale = scale.to(device)
        _t['im_detect'].tic()
        out = net(x)  # forward pass
        boxes, scores = detector.forward(out, self.priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]
        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        _t['misc'].tic()
        c_dets = np.empty([0, 5], dtype=np.float32)
        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                # c_dets = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = nms(c_dets, 0.45, force_cpu=(device != 'cuda'))
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets
        return all_boxes


def main():
    device = "cpu"
    img_path = "/home/workspace/jhwen/data/plate_data/check_off/JPEGImages/0000000000000000-181227-091400-091405-000007000220-sn00070.jpg"
    cfg = VOC_300
    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(device)
    img_dim = 300
    num_classes = 3
    net = build_net('test', img_dim, num_classes)  # initialize detector
    net.load_state_dict(torch.load("SSD_VGG_FPN_VOC_epoches_165.pth"))
    net.eval()
    net.to(device)
    top_k = 200
    detector = Detect(num_classes, 0, cfg)
    rgb_means = (104, 117, 123)
    opencv_image = cv2.imread(img_path)
    original_image = opencv_image.copy()
    all_boxes = detect(original_image, net, detector, device,
                       BaseTransform(net.size, rgb_means, (2, 0, 1)), top_k, thresh=0.2,
                       num_classes=num_classes)
    print(all_boxes)
    k = 1
    if len(all_boxes[k]) > 0:
        box_scores = all_boxes[k][:, 4]
        box_locations = all_boxes[k][:, 0:4]
        for i, box in enumerate(box_locations):
            box = [int(b) for b in box]
            p1 = (box[0], box[1])
            p2 = (box[2], box[3])
            cv2.rectangle(opencv_image, p1, p2, (0, 255, 0))
            title = "%s:%.2f" % (VOC_CLASSES[k], box_scores[i])
            p3 = (max(p1[0], 15), max(p1[1], 15))
            cv2.putText(opencv_image, title, p3, cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        save_file = os.path.join("result", "1" + '.jpg')
        print(save_file)
        cv2.imwrite(save_file, opencv_image)


if __name__ == '__main__':
    ssd_detector = SSDDetector()
    img_path = "/home/workspace/jhwen/data/plate_data/check_off/JPEGImages/0000000000000000-181227-091400-091405-000007000220-sn00070.jpg"
    img = cv2.imread(img_path)
    score, box = ssd_detector.detect(img)
    print(box)
