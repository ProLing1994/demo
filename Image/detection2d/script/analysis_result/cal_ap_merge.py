import argparse
import cv2
import numpy as np
import os
import sys
import pickle
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.detection2d.ssd_rfb_crossdatatraining.utils import nms
from Image.detection2d.script.analysis_result.cal_ap import draw_img, check_set_roi, parse_rec, voc_ap


def voc_eval(merge_class_name,
             detpath_dict,
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
        R = [obj for obj in recs[imagename] if obj['name'] in classname]
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

    # load det file
    detfile = detpath_dict[merge_class_name]
    if not os.path.isfile(detfile):

        det_dict = {}
        # nms for merge class name
        for class_idx in range(len(classname)):
            class_name_idx = classname[class_idx]
            detfile_idx = detpath_dict[class_name_idx]

            # load
            with open(detfile_idx, 'r') as f:
                lines_idx = f.readlines()

            splitlines = [x.strip().split('*') for x in lines_idx]  
            if len(splitlines):
                image_ids = [x[0] for x in splitlines]
                c_scores = np.array([float(x[1]) for x in splitlines])
                c_bboxes = np.array([[float(z) for z in x[2:]] for x in splitlines])
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)

                for image_idx in range(len(image_ids)):
                    image_name_idx = image_ids[image_idx]
                    if image_name_idx in det_dict:
                        det_dict[image_name_idx] = np.concatenate((det_dict[image_name_idx], c_dets[image_idx][np.newaxis, :]), axis=0)
                    else:
                        det_dict[image_name_idx] = c_dets[image_idx][np.newaxis, :]

        with open(detfile, 'w') as f:
            
            for image_name_idx in det_dict.keys():
                
                # nms
                keep = nms(det_dict[image_name_idx], 0.45)
                c_dets = det_dict[image_name_idx][keep, :]

                # the VOCdevkit expects 1-based indices
                for k in range(c_dets.shape[0]):
                    f.write('{:s}*{:.3f}*{:.1f}*{:.1f}*{:.1f}*{:.1f}\n'.
                            format(image_name_idx, c_dets[k, -1],
                                   c_dets[k, 0], c_dets[k, 1],
                                   c_dets[k, 2], c_dets[k, 3]))

    # read dets
    with open(detfile, 'r') as f:
        lines = f.readlines()
    
    # analysis 
    tp = np.zeros(0)
    fp = np.zeros(0)
    conf = np.zeros(0)
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
        # print("npos: {}".format(npos))
        if npos != 0:
            print("recall(TPR): {:.3f}({}/{})".format(tp.sum()/ float(npos), tp.sum(), npos))
        else:
            print("TPR: None")
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp_sum / np.maximum(tp_sum + fp_sum, np.finfo(np.float64).eps)
        print("precision(PPV): {:.3f}({}/{})".format(tp.sum()/ (tp.sum() + fp.sum()), tp.sum(), (tp.sum() + fp.sum())))
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    # draw img
    if write_bool:
        output_dir = os.path.join(cachedir, 'img_res_{}'.format(str(ovthresh)), merge_class_name)
        if width_height_ovthresh_bool:
            output_dir = os.path.join(os.path.dirname(output_dir) + '_{}'.format(str(width_ovthresh)), merge_class_name)
        if iou_uni_use_label_bool:
            output_dir = os.path.join(os.path.dirname(output_dir) + '_{}'.format("uniuselabel"), merge_class_name)
        if roi_set_bool:
            output_dir = os.path.join(os.path.dirname(output_dir) + '_{}'.format("roi"), merge_class_name)

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
                    if (img_width == 1920 and img_height == 1080) or (img_width == 1080 and img_height == 1920) or (img_width == 1280 and img_height == 720):
                        draw_img(R, img_path, output_img_path, roi_set_bool, roi_set_bbox_2M)
                    elif (img_width == 2592 and img_height == 1920) or (img_width == 1920 and img_height == 2592):
                        draw_img(R, img_path, output_img_path, roi_set_bool, roi_set_bbox_5M)
                    else:
                        raise InterruptedError
            
            if write_false_positive_bool:
                # 判断是否有假阳
                if R['fp_bool']:
                    if (img_width == 1920 and img_height == 1080) or (img_width == 1080 and img_height == 1920) or (img_width == 1280 and img_height == 720):
                        draw_img(R, img_path, output_img_path, roi_set_bool, roi_set_bbox_2M)
                    elif (img_width == 2592 and img_height == 1920) or (img_width == 1920 and img_height == 2592):
                        draw_img(R, img_path, output_img_path, roi_set_bool, roi_set_bbox_5M)
                    else:
                        raise InterruptedError

    return tp, fp, conf, npos, rec, prec, ap