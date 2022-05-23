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
    
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan/"    # trainval: 5846, train: 5261, val: 585, test: 650, all: 6496
    # args.input_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_5M/"    # trainval: 1445, train: 1300, val: 145, test: 161, all: 1606
    # args.input_dir =  "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/sandaofangxian/"    # trainval: 4476, train: 4028, val: 448, test: 498, all: 4972
    # args.input_dir =  "/yuanhuan/data/image/ZG_AHHBGS_detection/anhuihuaibeigaosu/"    # trainval: 9024, train: 8121, val: 903, test: 1003, all: 10027
    # args.input_dir =  "/mnt/huanyuan2/data/image/ZG_AHHBGS_detection/shenzhentiaoqiao/"    # trainval: 4538, train: 4084, val: 454, test: 505, all: 5043
    args.input_dir =  "/mnt/huanyuan2/data/image/ZG_AHHBGS_detection/anhuihuaibeigaosu_02/"    # trainval: 190, train: 171, val: 19, test: 22, all: 212

    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/daminghu/"    # trainval: 6485, train: 5836, val: 649, test: 721, all: 7206
    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/yongzou_night_hongwai/"    # trainval: 2681, train: 2412, val: 269, test: 298, all: 2979
    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/anquandao/"    # trainval: 2370, train: 2133, val: 237, test: 264, all: 2634
    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/anquandao_night_hongwai/"    # trainval: 2250, train: 2025, val: 225, test: 250, all: 2500
    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/shandongyingzikou/"    # trainval: 1970, train: 1773, val: 197, test: 219, all: 2189
    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/shandongyingzikou_night_diguangzhao/"    # trainval: 613, train: 551, val: 62, test: 69, all: 682
    # args.input_dir =  "/yuanhuan/data/image/ZG_BMX_detection/shenzhenlukou_yejian/"    # trainval: 6987, train: 6288, val: 699, test: 777, all: 7764

    args.trainval_file = args.input_dir + "ImageSets/Main/trainval.txt"
    args.train_file = args.input_dir + "ImageSets/Main/train.txt"
    args.val_file = args.input_dir + "ImageSets/Main/val.txt"
    args.test_file = args.input_dir + "ImageSets/Main/test.txt"

    args.test_size = 0.1
    args.val_size = 0.1

    args.jpg_dir =  args.input_dir + "JPEGImages/"
    args.xml_dir =  args.input_dir + "XML/"

    split(args)
