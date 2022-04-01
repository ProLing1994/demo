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


def cv_plot_rectangle(img, bbox, color=None, mode='xywh', thickness=3):
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

    return objects


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
             roi_set_bbox=[0, 0, 1920, 1024],
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
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] in classname]

        # 是否设置 roi 区域，忽略边缘区域
        if roi_set_bool:
            bbox = np.array([x['bbox'] for x in R if check_set_roi(x['bbox'], roi_set_bbox) ])
            difficult = np.array([x['difficult'] for x in R if check_set_roi(x['bbox'], roi_set_bbox) ]).astype(np.bool)
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

    # load
    with open(detfile, 'r') as f:
            lines = f.readlines()
    
    # analysis 
    ntp = 0
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

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)

            # 是否设置 roi 区域，忽略边缘区域
            if roi_set_bool:
                if not check_set_roi(bb, roi_set_bbox):
                    continue

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
                        ntp += 1
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                        R['fp_bool'] = True
            else:
                fp[d] = 1.
                R['fp_bool'] = True
            
            R['det_res'].append(bb)

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        print("npos: {}".format(npos))
        if npos != 0:
            print("tpr: {:.3f}({}/{})".format(ntp/ float(npos), ntp, npos))
        else:
            print("tpr: None")
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    # draw img
    if args.write_bool:
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

            img_path = os.path.join(args.jpg_dir, imagename + '.jpg')
            output_img_path = os.path.join(output_dir, imagename + '.jpg')

            if args.write_unmatched_bool:
                # 判断是否有漏检
                if not len(R['det']) == np.array( R['det'] ).sum():
                     draw_img(R, img_path, output_img_path, roi_set_bool, roi_set_bbox)
            
            if args.write_false_positive_bool:
                # 判断是否有假阳
                if R['fp_bool']:
                    draw_img(R, img_path, output_img_path, roi_set_bool, roi_set_bbox)

    return rec, prec, ap


def calculate_ap(args):
    anno_path = os.path.join(args.anno_dir, '%s.xml')
    cache_dir = args.output_dir
    
    # clear
    cache_file = os.path.join(cache_dir, 'annots.pkl')
    if os.path.exists(cache_file):
        os.remove(cache_file)

    aps = []
    for merge_class_name in args.merge_ap_dict.keys():
        rec, prec, ap = voc_eval(
            merge_class_name=merge_class_name,
            detpath_dict=args.det_path_dict, 
            annopath=anno_path, 
            imagesetfile=args.imageset_file, 
            classname=args.merge_ap_dict[merge_class_name], 
            cachedir=cache_dir,
            ovthresh=args.over_thresh, 
            iou_uni_use_label_bool=args.iou_uni_use_label_bool,
            width_height_ovthresh_bool=args.width_height_over_thresh_bool,
            width_ovthresh=args.width_over_thresh,
            height_ovthresh=args.height_over_thresh,
            roi_set_bool=args.roi_set_bool,
            roi_set_bbox=args.roi_set_bbox,
            use_07_metric=args.use_07_metric)

        aps += [ap]
        print('AP for {} = {:.3f} \n'.format(merge_class_name, ap))
    
    print('Mean AP = {:.3f}'.format(np.mean(aps)))

    return 

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    #####################################
    # Car_Bus_Truck_Licenseplate
    # 测试集图像
    #####################################
    # args.data_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan/"
    # args.imageset_file = os.path.join(args.data_dir, "ImageSets/Main/test.txt")
    # # args.anno_dir =  os.path.join(args.data_dir, "Annotations_CarBusTruckLicenseplate_w_height/")                # 高度大于 24 的 清晰车牌
    # # args.anno_dir =  os.path.join(args.data_dir, "Annotations_CarBusTruckLicenseplate_w_fuzzy_w_height/")        # 高度大于 24 的 清晰车牌 & 模糊车牌
    # # args.anno_dir =  os.path.join(args.data_dir, "Annotations_CarBusTruckLicenseplate/")                         # 清晰车牌
    # args.anno_dir =  os.path.join(args.data_dir, "Annotations_CarBusTruckLicenseplate_w_fuzzy/")                 # 清晰车牌 & 模糊车牌
    # args.jpg_dir =  os.path.join(args.data_dir,  "JPEGImages/")
    # # args.input_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-02-24-15_focalloss_4class_car_bus_truck_licenseplate_zg_w_fuzzy_plate/eval_epoches_299/ZG_ZHJYZ_detection_jiayouzhan_test/results/"
    # args.input_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-03-09-17_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/eval_epoches_299/ZG_ZHJYZ_detection_jiayouzhan_test/results/"

    ######################################
    # 测试集：
    ######################################
    args.data_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/"
    args.imageset_file = os.path.join(args.data_dir, "AHHBAS_41c/images.txt")
    # args.anno_dir =  os.path.join(args.data_dir, "AHHBAS_41c_Annotations_CarBusTruckLicenseplate_w_height/")               # 高度大于 24 的 清晰车牌
    # args.anno_dir =  os.path.join(args.data_dir, "AHHBAS_41c_Annotations_CarBusTruckLicenseplate_w_fuzzy_w_height/")       # 高度大于 24 的 清晰车牌 & 模糊车牌
    # args.anno_dir =  os.path.join(args.data_dir, "AHHBAS_41c_Annotations_CarBusTruckLicenseplate/")                        # 清晰车牌
    args.anno_dir =  os.path.join(args.data_dir, "AHHBAS_41c_Annotations_CarBusTruckLicenseplate_w_fuzzy/")                # 清晰车牌 & 模糊车牌
    args.jpg_dir =  os.path.join(args.data_dir,  "AHHBAS_41c/")
    # args.input_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-02-24-15_focalloss_4class_car_bus_truck_licenseplate_zg_w_fuzzy_plate/eval_epoches_299/jiayouzhan_test_image_AHHBAS_41c/results/"
    args.input_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-03-09-17_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/eval_epoches_299/jiayouzhan_test_image_AHHBAS_41c/results/"

    args.merge_ap_dict = { 'car': ['car'], 'bus_truck': ['bus', 'truck'], 'car_bus_truck': ['car', 'bus', 'truck'] }
    # args.merge_ap_dict = { 'bus_truck': ['bus', 'truck'] }

    args.det_path_dict = { 'car': args.input_dir + 'det_test_car.txt',
                           'bus': args.input_dir + 'det_test_bus.txt',
                           'truck': args.input_dir + 'det_test_truck.txt',
                           'car_bus_truck': args.input_dir + 'det_test_car_bus_truck.txt', 
                           'bus_truck': args.input_dir + 'det_test_bus_truck.txt', 
                         } 

    args.over_thresh = 0.4
    args.use_07_metric = False

    # 是否设置 roi 区域，忽略边缘区域
    args.roi_set_bool = False
    # args.roi_set_bbox = [200, 110, 1720, 970]       # 2M
    args.roi_set_bbox = [300, 150, 2292, 1770]       # 5M
    
    # 是否在计算 iou 的过程中，计算 uni 并集的面积只关注 label 的面积
    args.iou_uni_use_label_bool = False

    # 是否关注车牌横向iou结果
    args.width_height_over_thresh_bool = False
    args.width_over_thresh = 0.9
    args.height_over_thresh = 0.0

    # 是否保存识别结果和检出结果
    args.write_bool = True

    # 是否保存漏检结果
    args.write_unmatched_bool = True

    # 是否保存假阳结果
    args.write_false_positive_bool = True

    args.output_dir = args.input_dir

    calculate_ap(args)