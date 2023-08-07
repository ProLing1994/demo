"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import argparse
import os

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
import torch
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')

# from SSD_model.fpn_rfb_liujun import build_net
# from Hungarian import Hungarian


def cv2_demo(net, transform, image_path):
    # orig_img = cv2.imread(image_path)
    orig_img = image_path

    height, width = orig_img.shape[:2]
    x = torch.from_numpy(transform(orig_img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))

    y = net(x)

    res = []

    detections = y.data

    scale = torch.Tensor([width, height, width, height])

    for i in range(1, detections.size(1)):

        j = 0
        while detections[0, i, j, 0] > 0.4:
            pt = (detections[0, i, j, 1:] * scale)
            xmin = int(pt[0]) if int(pt[0]) >= 0 else 0
            ymin = int(pt[1]) if int(pt[1]) >= 0 else 0
            xmax = int(pt[2]) if int(pt[2]) <= 1920 else 1920
            ymax = int(pt[3]) if int(pt[3]) <= 1080 else 1080

            if xmin >= xmax or ymin >= xmax:
                j += 1
                continue

            res.append(xmin)
            res.append(ymin)
            res.append(xmax)
            res.append(ymax)
            res.append(i)

            cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            ptext = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
            title = str(i) + ":%.2f" % (detections[0, i, j, 0])
            cv2.putText(orig_img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 1/2, (0, 0, 255), 2, cv2.LINE_AA)
            j += 1

    # cv2.imwrite('./test_result1.jpg', orig_img)
    # cv2.imshow("test", orig_img)
    # cv2.waitKey(0)

    return np.array(res).reshape(-1, 5)


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels


def linear_assignment(cost_matrix):
    # try:
    #     import lap
    #     _, x, y = lap.lapjv(cost_matrix, extend_cost=True)

    #     return np.array([[y[i], i] for i in x if i >= 0])  #
    # except ImportError:

    #     from scipy.optimize import linear_sum_assignment

    #     x, y = linear_sum_assignment(cost_matrix)

    #     return np.array(list(zip(x, y)))

    from scipy.optimize import linear_sum_assignment

    x, y = linear_sum_assignment(cost_matrix)

    return np.array(list(zip(x, y)))


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):

    xmin = xmin1 if xmin1 > xmin2 else xmin2
    ymin = ymin1 if ymin1 > ymin2 else ymin2

    xmax = xmax1 if xmax1 < xmax2 else xmax2
    ymax = ymax1 if ymax1 < ymax2 else ymax2

    interWidth = xmax - xmin
    interHeight = ymax - ymin

    interWidth = interWidth if interWidth > 0 else 0
    interHeight = interHeight if interHeight > 0 else 0

    interArea = interWidth * interHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    totalArea = area1 + area2 - interArea

    return interArea / totalArea


def BATCH_IOU(bb_test, bb_gt, iou_threshold):
    bb_test1 = bb_test.copy()

    iou_res = []
    index_matrix = []
    for i in range(len(bb_test1)):
        for j in range(len(bb_gt)):
            obj1 = bb_test1[i]
            obj2 = bb_gt[j]
            iou = IOU(obj1[0], obj1[1], obj1[2], obj1[3], obj2[0], obj2[1], obj2[2], obj2[3])
            if iou > iou_threshold:
                iou_res.append(iou)
                index_matrix.append(i)
                index_matrix.append(j)
            else:
                iou_res.append(iou)

    # temp = len(bb_test1) if len(bb_test1) > len(bb_gt) else len(bb_gt)
    return np.array(iou_res).reshape(-1, len(bb_gt)), np.array(index_matrix).reshape(-1, 2)


def iou_batch(bb_test, bb_gt, iou_threshold):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    iou_res, index_matrix = BATCH_IOU(bb_test, bb_gt, iou_threshold)

    return iou_res, index_matrix

    # bb_gt = np.expand_dims(bb_gt, 0)
    # bb_test = np.expand_dims(bb_test, 1)
    #
    # xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    # yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    # xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    # yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    #
    # w = np.maximum(0., xx2 - xx1)
    # h = np.maximum(0., yy2 - yy1)
    # wh = w * h
    #
    # o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    #
    # return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        # return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        return np.empty((0, 2), dtype=int), np.arange(len(detections)) #, np.empty((0, 5), dtype=int)

    iou_matrix, index_matrix = iou_batch(detections, trackers, 0.3)
    # print(index_matrix)

    # matched_indices:是一个匹配矩阵，第一列是匹配到的检测目标序号，第二列是匹配到的跟踪目标序号
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        # print(a)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            # matched_indices = np.stack(np.where(a), axis=1)
            matched_indices = index_matrix
        else:
            matched_indices = linear_assignment(-iou_matrix)


            # print(iou_matrix.shape)
            # hungarian = Hungarian(-iou_matrix)
            # hungarian.calculate()
            # result = hungarian.get_results()
            # matched_indices = np.array(result)
            # print('1:', matched_indices)

    else:
        matched_indices = np.empty(shape=(0, 2))


    # print(matched_indices)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)

    # unmatched_trackers = []
    # for t, trk in enumerate(trackers):
    #     if (t not in matched_indices[:, 1]):
    #         unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            # unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections) #, np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))

        to_del = []
        ret = []
        # for t, trk in enumerate(trks):
        for t in range(len(trks)):
            trk = trks[t]
            pos = self.trackers[t].predict()[0]

            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

            if np.any(np.isnan(pos)):
                to_del.append(t)

        #trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)

        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1

            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=5)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('This is main ...')

    args = parse_args()
    phase = args.phase

    mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)  # create instance of the SORT tracker

    source_videos_path = r"D:/Data/testvideos/test"
    video_names = os.listdir(source_videos_path)

    net = build_net('test', 300, 4)
    net.load_state_dict(
        torch.load('./SSD_model/SSD_VGG_FPN_RFB_VOC_epoches_190_0806.pth', map_location={'cuda:1': 'cuda:0'}))
    net.eval()
    transform = BaseTransform(300, (104, 117, 123))

    for name in video_names:
        videopath = os.path.join(source_videos_path, name)

        cap = cv2.VideoCapture(videopath)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        success, frame = cap.read()
        while success:

            dets = cv2_demo(net, transform, frame)

            trackers = mot_tracker.update(dets)

            for d in trackers:
                d = d.astype(np.int32)
                cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), color=(0, 0, 255), thickness=2)
                cv2.putText(frame, "%d" % (d[4] % 32), (d[0], d[1] - 2), 0, 1/2, [225, 0, 255], thickness=2, lineType=cv2.LINE_AA)
            cv2.imshow("sort", frame)
            cv2.waitKey(10)
            #
            success, frame = cap.read()
        cap.release
