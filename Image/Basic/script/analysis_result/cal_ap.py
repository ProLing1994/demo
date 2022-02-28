import argparse
import numpy as np
import os
import pickle
import xml.etree.ElementTree as ET


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


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
    detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
    annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
    (default True)
    """
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
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    ntp = 0
    with open(detfile, 'r') as f:
        lines = f.readlines()
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
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        ntp += 1
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        print("tpr: {:.3f}".format(ntp/ float(npos)))
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def calculate_ap(args):
    anno_path = os.path.join(args.anno_dir, '%s.xml')
    cache_dir = args.output_dir

    aps = []
    for class_name in args.det_path_dict.keys():
        rec, prec, ap = voc_eval(
            detpath=args.det_path_dict[class_name], 
            annopath=anno_path, 
            imagesetfile=args.imageset_file, 
            classname=class_name, 
            cachedir=cache_dir,
            ovthresh=args.over_thresh, 
            use_07_metric=args.use_07_metric)

        aps += [ap]
        print('AP for {} = {:.4f} \n'.format(class_name, ap))
    
    print('Mean AP = {:.4f}'.format(np.mean(aps)))

    return 

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Car_Licenseplate
    # args.data_dir = "/yuanhuan/data/image/LicensePlate/China/"
    # args.imageset_file = args.data_dir + "ImageSets/Main/test.txt"
    # args.anno_dir =  args.data_dir + "Annotations_CarLicenseplate/"

    # # # ssd rfb
    # # args.det_path_dict = { 'car': '/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-02-11-16_focalloss_car_licenseplate/eval_epoches_299/LicensePlate_China_test/results/det_test_car.txt',
    # #                        'license_plate': '/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-02-11-16_focalloss_car_licenseplate/eval_epoches_299/LicensePlate_China_test/results/det_test_license_plate.txt',
    # #                      } 
    # # args.over_thresh = 0.4
    # # args.use_07_metric = False
    # # args.output_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-02-11-16_focalloss_car_licenseplate/eval_epoches_299/LicensePlate_China_test/results/"

    # # yolox
    # args.det_path_dict = { 'car': '/yuanhuan/model/image/yolox_vgg/yoloxv2_vggrm_640_384_car_license_plate/eval_epoches_24/LicensePlate_China_xml/det_test_car.txt',
    #                        'license_plate': '/yuanhuan/model/image/yolox_vgg/yoloxv2_vggrm_640_384_car_license_plate/eval_epoches_24/LicensePlate_China_xml/det_test_license_plate.txt',
    #                      } 
    # # args.over_thresh = 0.35
    # args.over_thresh = 0.4
    # args.use_07_metric = False
    # args.output_dir = "/yuanhuan/model/image/yolox_vgg/yoloxv2_vggrm_640_384_car_license_plate/eval_epoches_24/LicensePlate_China_xml/"

    # Car_Bus_Truck_Licenseplate
    args.data_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan/"
    args.imageset_file = os.path.join(args.data_dir, "ImageSets/Main/test.txt")
    args.anno_dir =  os.path.join(args.data_dir, "Annotations_CarBusTruckLicenseplate_w_fuzzy/")

    # ssd rfb
    args.det_path_dict = { 'car': '/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-02-24-15_focalloss_4class_car_bus_truck_licenseplate_zg_w_fuzzy_plate/eval_epoches_299/ZG_ZHJYZ_detection_jiayouzhan_test/results/det_test_car.txt',
                           'bus': '/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-02-24-15_focalloss_4class_car_bus_truck_licenseplate_zg_w_fuzzy_plate/eval_epoches_299/ZG_ZHJYZ_detection_jiayouzhan_test/results/det_test_bus.txt',
                           'truck': '/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-02-24-15_focalloss_4class_car_bus_truck_licenseplate_zg_w_fuzzy_plate/eval_epoches_299/ZG_ZHJYZ_detection_jiayouzhan_test/results/det_test_truck.txt',
                           'license_plate': '/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-02-24-15_focalloss_4class_car_bus_truck_licenseplate_zg_w_fuzzy_plate/eval_epoches_299/ZG_ZHJYZ_detection_jiayouzhan_test/results/det_test_license_plate.txt',
                         } 
    args.over_thresh = 0.4
    args.use_07_metric = False
    args.output_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-02-24-15_focalloss_4class_car_bus_truck_licenseplate_zg_w_fuzzy_plate/eval_epoches_299/ZG_ZHJYZ_detection_jiayouzhan_test/results/"

    calculate_ap(args)