import argparse
import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm
import xml.etree.ElementTree as ET


color_dict = {
                "lable_matched": (0, 0, 255), 
                "lable_unmatched": (0, 255, 0), 
                "det": (0, 255, 255), 
                "roi": (255, 255, 255), 
            }

def draw_img(R, img_path, output_img_path, roi_bool, roi_bbox):

    img = cv2.imread(img_path)

    # 绘制 det bbox
    for det_bbox_idx in range(len(R['det_res'])):
        det_bbox = R['det_res'][det_bbox_idx]
        img = cv_plot_rectangle(img, det_bbox, mode='ltrb', color=color_dict["det"])
    
    # 绘制 label bbox
    for label_bbox_idx in range(len(R['bbox'])):
        label_bbox = R['bbox'][label_bbox_idx]
        det_mark = R['det'][label_bbox_idx]
        
        if det_mark:
            img = cv_plot_rectangle(img, label_bbox, mode='ltrb', color=color_dict["lable_matched"])
        else:
            img = cv_plot_rectangle(img, label_bbox, mode='ltrb', color=color_dict["lable_unmatched"])
    
    # 绘制 roi 区域
    if roi_bool:
        img = cv_plot_rectangle(img, roi_bbox, mode='ltrb', color=color_dict["roi"])

    cv2.imwrite(output_img_path, img)


def cv_plot_rectangle(img, bbox, mode='xywh', color=None, thickness=2):
    if color is None:
        color = color_dict["det"]
    if mode == 'xywh':
        x, y, w, h = bbox
        xmin, ymin, xmax, ymax = x, y, w + x, h + y
    elif mode == 'ltrb':
        xmin, ymin, xmax, ymax = bbox
    else:
        print("Unknown plot mode")
        return None
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    img_p = img.copy()
    return cv2.rectangle(img_p, (xmin, ymin),
                         (xmax, ymax), color=color, thickness=thickness)


def check_set_roi(in_box, roi_bbox):
    roi_bool = False

    if in_box[2] > roi_bbox[0] and in_box[0] < roi_bbox[2] and in_box[3] > roi_bbox[1] and in_box[1] < roi_bbox[3]:
        roi_bool = True
    
    return roi_bool


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    img_width = tree.find('size').find('width').text
    img_height = tree.find('size').find('height').text
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects, img_width, img_height


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             iou_uni_use_label_bool=False,
             width_height_ovthresh_bool=False,
             width_ovthresh=0.5,
             height_ovthresh=0.5,
             roi_set_bool=False,
             roi_set_bbox_2M=[0, 0, 1920, 1024],
             roi_set_bbox_5M=[0, 0, 2592, 1920],
             write_bool=False,
             jpg_dir=None,
             write_unmatched_bool=False,
             write_false_positive_bool=False,
             use_07_metric=True):
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # save the truth data as pickle,if the pickle in the file, just load it.
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        img_widths = {}
        img_heights = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename], img_widths[imagename], img_heights[imagename] = parse_rec(annopath % (imagename))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump((recs, img_widths, img_heights), f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs, img_widths, img_heights = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        img_width = int(img_widths[imagename])
        img_height = int(img_heights[imagename])

        # 是否设置 roi 区域，忽略边缘区域
        if roi_set_bool:
            if (img_width == 1920 and img_height == 1080) or (img_width == 1080 and img_height == 1920):
                bbox = np.array([x['bbox'] for x in R if check_set_roi(x['bbox'], roi_set_bbox_2M) ])
                difficult = np.array([x['difficult'] for x in R if check_set_roi(x['bbox'], roi_set_bbox_2M) ]).astype(np.bool)
            elif (img_width == 2592 and img_height == 1920) or (img_width == 1920 and img_height == 2592):
                bbox = np.array([x['bbox'] for x in R if check_set_roi(x['bbox'], roi_set_bbox_5M) ])
                difficult = np.array([x['difficult'] for x in R if check_set_roi(x['bbox'], roi_set_bbox_5M) ]).astype(np.bool)
            else:
                raise InterruptedError
        else:
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(bbox)
        npos = npos + sum(~difficult)
        det_res = []
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det,
                                 'det_res': det_res,
                                 'fp_bool': False,
                                 }

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    
    # analysis 
    if any(lines) == 1:
        # lines: CH10-20210615-142026-1429200000010872*0.563*386.8*1382.6*752.2*1548.1
        splitlines = [x.strip().split('*') for x in lines]  
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        confidence = [confidence[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        conf = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            img_width = int(img_widths[image_ids[d]])
            img_height = int(img_heights[image_ids[d]])
            bb = BB[d, :].astype(float)

            # 是否设置 roi 区域，忽略边缘区域
            if roi_set_bool:
                if (img_width == 1920 and img_height == 1080) or (img_width == 1080 and img_height == 1920):
                    if not check_set_roi(bb, roi_set_bbox_2M):
                        continue
                elif (img_width == 2592 and img_height == 1920) or (img_width == 1920 and img_height == 2592):
                    if not check_set_roi(bb, roi_set_bbox_5M):
                        continue
                else:
                    raise InterruptedError

            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                if iou_uni_use_label_bool:
                    uni = (BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1])
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

                # compute overlaps in width $ height
                if width_height_ovthresh_bool:
                    # width_overlaps
                    iymin = np.maximum(0, 0)
                    iymax = np.minimum(1, 1)
                    ih_temp = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih_temp
                    uni = ((bb[2] - bb[0]) * (1 - 0) +
                        (BBGT[:, 2] - BBGT[:, 0]) *
                        (1 - 0) - inters)
                    if iou_uni_use_label_bool:
                        uni = (BBGT[:, 2] - BBGT[:, 0]) * (1 - 0)
                    width_overlaps = inters / uni
                
                    # height_overlaps
                    ixmin = np.maximum(0, 0)
                    ixmax = np.minimum(1, 1)
                    iw_temp = np.maximum(ixmax - ixmin, 0.)
                    inters = iw_temp * ih
                    uni = ((1 - 0) * (bb[3] - bb[1]) +
                        (1 - 0) *
                        (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    if iou_uni_use_label_bool:
                        uni =  (1 - 0) * (BBGT[:, 3] - BBGT[:, 1])
                    height_overlaps = inters / uni

            if width_height_ovthresh_bool:
                # tp_bool = ovmax > ovthresh and width_overlaps[jmax] > width_ovthresh
                tp_bool = ovmax > ovthresh and width_overlaps[jmax] > width_ovthresh and height_overlaps[jmax] > height_ovthresh
            else:
                tp_bool = ovmax > ovthresh

            if tp_bool: 
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                        R['fp_bool'] = True
            else:
                fp[d] = 1.
                R['fp_bool'] = True
            
            conf[d] = confidence[d]
            R['det_res'].append(bb)

        # compute precision recall
        fp_sum = np.cumsum(fp)
        tp_sum = np.cumsum(tp)
        rec = tp_sum / float(npos)
        print("npos: {}".format(npos))
        if npos != 0:
            print("tpr: {:.3f}({}/{})".format(tp.sum()/ float(npos), tp.sum(), npos))
        else:
            print("tpr: None")
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp_sum / np.maximum(tp_sum + fp_sum, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    # draw img
    if write_bool:
        output_dir = os.path.join(cachedir, 'img_res_{}'.format(str(ovthresh)), classname)
        if width_height_ovthresh_bool:
            output_dir = os.path.join(os.path.dirname(output_dir) + '_{}_{}'.format(str(width_ovthresh), str(height_ovthresh)), classname)
        if iou_uni_use_label_bool:
            output_dir = os.path.join(os.path.dirname(output_dir) + '_{}'.format("uniuselabel"), classname)
        if roi_set_bool:
            output_dir = os.path.join(os.path.dirname(output_dir) + '_{}'.format("roi"), classname)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for imagename in tqdm(imagenames):
            R = class_recs[imagename]
            img_width = int(img_widths[imagename])
            img_height = int(img_heights[imagename])

            img_path = os.path.join(jpg_dir, imagename + '.jpg')
            output_img_path = os.path.join(output_dir, imagename + '.jpg')

            if write_unmatched_bool:
                # 判断是否有漏检
                if not len(R['det']) == np.array( R['det'] ).sum():
                    if (img_width == 1920 and img_height == 1080) or (img_width == 1080 and img_height == 1920):
                        draw_img(R, img_path, output_img_path, roi_set_bool, roi_set_bbox_2M)
                    elif (img_width == 2592 and img_height == 1920) or (img_width == 1920 and img_height == 2592):
                        draw_img(R, img_path, output_img_path, roi_set_bool, roi_set_bbox_5M)
                    else:
                        raise InterruptedError
            if write_false_positive_bool:
                # 判断是否有假阳
                if R['fp_bool']:
                    if (img_width == 1920 and img_height == 1080) or (img_width == 1080 and img_height == 1920):
                        draw_img(R, img_path, output_img_path, roi_set_bool, roi_set_bbox_2M)
                    elif (img_width == 2592 and img_height == 1920) or (img_width == 1920 and img_height == 2592):
                        draw_img(R, img_path, output_img_path, roi_set_bool, roi_set_bbox_5M)
                    else:
                        raise InterruptedError

    return tp, fp, conf, npos, rec, prec, ap