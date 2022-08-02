import argparse
import numpy as np
import os

from sklearn.model_selection import train_test_split


def split(args):

    # mkdir 
    if not os.path.exists(os.path.dirname(args.trainval_file)):
        os.makedirs(os.path.dirname(args.trainval_file))

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]

    trainval_list, test_list = train_test_split(jpg_list, test_size=args.test_size, random_state=0)
    train_list, val_list = train_test_split(trainval_list, test_size=args.val_size, random_state=0)

    print("length: trainval: {}, train: {}, val: {}, test: {}, all: {}".format(len(trainval_list), len(train_list), len(val_list), len(test_list), (len(train_list) + len(val_list) + len(test_list))))
    with open(args.trainval_file, "w") as f:
        for jpg in trainval_list:
            f.write(jpg.replace(".jpg", ""))
            f.write("\n")

    with open(args.test_file, "w") as f:
        for jpg in test_list:
            f.write(jpg.replace(".jpg", ""))
            f.write("\n")

    with open(args.train_file, "w") as f:
        for jpg in train_list:
            f.write(jpg.replace(".jpg", ""))
            f.write("\n")

    with open(args.val_file, "w") as f:
        for jpg in val_list:
            f.write(jpg.replace(".jpg", ""))
            f.write("\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.input_dir = "/yuanhuan/data/image/LicensePlate/China/"             # trainval: 35396, train: 31856, val: 3540, test: 3933
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/China_6mm/"         # trainval: 1762, train: 1585, val: 177, test: 196
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/Europe/"            # trainval: 18644, train: 16779, val: 1865, test: 2072
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/Mexico/"            # trainval: 10578, train: 9520, val: 1058, test: 1176
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone/"      # trainval: 102242, train: 92017, val: 10225, test: 11361
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_new/"  # trainval: 8840, train: 7956, val: 884, test: 983, all: 9823
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate/"  # trainval: 13485, train: 12136, val: 1349, test: 1499, all: 14984
    # args.input_dir = "/yuanhuan/data/image/RM_BSD/bsd_20220425_20220512/"   # trainval: 13317, train: 11985, val: 1332, test: 1480, all: 14797

    args.input_dir = "/yuanhuan/data/image/ZF_Europe/england/"              # trainval: 33597, train: 30237, val: 3360, test: 3733, all: 37330
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/england_1080p/"        # trainval: 22255, train: 20029, val: 2226, test: 2473, all: 24728
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/france/"               # trainval: 39950, train: 35955, val: 3995, test: 4439, all: 44389
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/italy/"                # trainval: 6088, train: 5479, val: 609, test: 677, all: 6765
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/netherlands/"          # trainval: 32608, train: 29347, val: 3261, test: 3624, all: 36232
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/moni/"                 # trainval: 7582, train: 6823, val: 759, test: 843, all: 8425
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/moni_0415/"            # trainval: 9661, train: 8694, val: 967, test: 1074, all: 10735
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/hardNeg/"              # trainval: 17540, train: 15786, val: 1754, test: 1949, all: 19489

    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan/"    # trainval: 5846, train: 5261, val: 585, test: 650, all: 6496
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_5M/"    # trainval: 1445, train: 1300, val: 145, test: 161, all: 1606
    # args.input_dir =  "/yuanhuan/data/image/ZG_ZHJYZ_detection/sandaofangxian/"    # trainval: 5060, train: 4553, val: 507, test: 563, all: 5623
    # args.input_dir =  "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu/"    # trainval: 18045, train: 16239, val: 1806, test: 2007, all: 20052
    # args.input_dir =  "/yuanhuan/data/image/ZG_ZHJYZ_detection/shenzhentiaoqiao/"    # trainval: 4538, train: 4084, val: 454, test: 505, all: 5043
    # args.input_dir =  "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu_night_diguangzhao/"    # trainval: 23315, train: 20983, val: 2332, test: 2591, all: 25906

    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/daminghu/"    # trainval: 9218, train: 8296, val: 922, test: 1025, all: 10243
    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/daminghu_night/"    # trainval: 2360, train: 2124, val: 236, test: 263, all: 2623
    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/shandongyingzikou/"    # trainval: 1970, train: 1773, val: 197, test: 219, all: 2189
    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/shandongyingzikou_night_hongwai/"    # trainval: 613, train: 551, val: 62, test: 69, all: 682
    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/yongzou_night_hongwai/"    # trainval: 2681, train: 2412, val: 269, test: 298, all: 2979
    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/shenzhenlukou/"    # trainval: 1264, train: 1137, val: 127, test: 141, all: 1405
    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/shenzhenlukou_night_hongwai/"    # trainval: 3427, train: 3084, val: 343, test: 381, all: 3808
    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/shenzhenlukou_night_diguangzhao/"    # trainval: 2295, train: 2065, val: 230, test: 256, all: 2551
    # # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/rongheng/"    # trainval: 4619, train: 4157, val: 462, test: 514, all: 5133
    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/rongheng_night_hongwai/"    # trainval: 4499, train: 4049, val: 450, test: 500, all: 4999

    # args.input_dir = "/yuanhuan/data/image/Open_Source/MOT/MOT17/"    # trainval: 4784, train: 4305, val: 479, test: 532, all: 5316
    # args.input_dir = "/yuanhuan/data/image/Open_Source/MOT/MOT20/"    # trainval: 8037, train: 7233, val: 804, test: 894, all: 8931
    # args.input_dir = "/yuanhuan/data/image/Open_Source/NightOwls/nightowls/"    # trainval: 16332, train: 14698, val: 1634, test: 1815, all: 18147
    # args.input_dir = "/yuanhuan/data/image/Open_Source/Cityscapes/cityscapes/"    # trainval: 3111, train: 2799, val: 312, test: 346, all: 3457

    args.trainval_file = args.input_dir + "ImageSets/Main/trainval.txt"
    args.train_file = args.input_dir + "ImageSets/Main/train.txt"
    args.val_file = args.input_dir + "ImageSets/Main/val.txt"
    args.test_file = args.input_dir + "ImageSets/Main/test.txt"

    args.test_size = 0.1
    args.val_size = 0.1

    args.jpg_dir =  args.input_dir + "JPEGImages/"
    args.xml_dir =  args.input_dir + "XML/"

    split(args)
