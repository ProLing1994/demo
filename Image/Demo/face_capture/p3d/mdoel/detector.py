import cv2
import numpy as np
import sys

caffe_root = '/home/huanyuan/code/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Demo.face_capture.p3d.mdoel.nms import *
from Image.Demo.face_capture.p3d.mdoel.prior_box_cpu import *


class SSDDetector(object):

    def __init__(self, prototxt, model_path):
        
        self.prototxt = prototxt
        self.model_path = model_path

        # params
        self.cfg =  {
                        'feature_maps' : [38, 19, 10, 5, 3, 1],
                        'min_dim' : 300,
                        'steps' : [8, 16, 32, 64, 100, 300],
                        'min_sizes' : [20, 60, 111, 162, 213, 264],
                        'max_sizes' : [60, 111, 162, 213, 264, 315],
                        'aspect_ratios' : [[2, 3], [2, 3], [2, 3], [2, 3], [2,3], [2,3]],
                        'variance' : [0.1, 0.2],
                        'clip' : True,
                        'anchor_level_k' : [9,9,9,9,9,9]
                    }
        self.name_classes = ('background', 'person')
        self.num_classes = 2
        self.img_dim = 300
        self.rgb_means = (104, 117, 123)
        self.threshold = 0.4
        
        self.model_init()
    
    def model_init(self):
        # ssd
        self.net = caffe.Net(self.prototxt, self.model_path, caffe.TEST)

        # ssd
        priorbox = PriorBox(self.cfg)
        self.priors = priorbox.forward()

    def detect(self, img_ori):

        def preprocess(src):
            img = cv2.resize(src, (self.img_dim, self.img_dim)).astype(np.float32)
            rgb_mean = np.array(self.rgb_means, dtype=np.int)
            img -= rgb_mean
            img = img.astype(np.float32)
            return img

        def decode(loc, priors, variances):
            """Decode locations from predictions using priors to undo
            the encoding we did for offset regression at train time.
            Args:
                loc: location predictions for loc layers,
                    Shape: [num_priors,4]
                priors: Prior boxes in center-offset form.
                    Shape: [num_priors,4].
                variances: (list[float]) Variances of priorboxes
            Return:
                decoded bounding box predictions
            """

            boxes = np.concatenate((
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] += boxes[:, :2]
            #print(boxes)
            return boxes

        def postprocess(img, out):
            h = img.shape[0]
            w = img.shape[1]
            boxes = out['mbox_loc_reshape'][0]
            boxes = decode(boxes, self.priors, self.cfg['variance'])
            boxes *= np.array([w, h, w, h])
            scores = out['mbox_conf_reshape'][0]
            return (boxes.astype(np.int32), scores)

        # forward
        img = preprocess(img_ori)
        img = img.transpose((2, 0, 1))
        self.net.blobs['data'].data[...] = img
        out = self.net.forward()
        boxes, scores = postprocess(img_ori, out)

        # dets
        all_boxes = [[] for _ in range(self.num_classes)]

        # nms
        for j in range(1, self.num_classes):
        
            inds = np.where(scores[:, j] > self.threshold)[0]
            if len(inds) == 0:
                all_boxes[j] = np.empty([0, 5], dtype=np.float32)
                continue

            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, 0.45)
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets
        
        # out_dict
        h, w, _ = img_ori.shape
        out_dict = {}
        for k in range(1, self.num_classes):
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
                    bbox_out.append([l, t, r, b, score])
                box_locations = bbox_out

                out_dict[self.name_classes[k]] = box_locations
        return out_dict