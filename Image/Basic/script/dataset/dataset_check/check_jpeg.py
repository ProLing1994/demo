import argparse
# import cv2
import numpy as np
import os
from tqdm import tqdm


def check_jepg(args):
    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]

    for idx in tqdm(range(len(jpg_list))):
        jpg_path = os.path.join(args.jpg_dir, jpg_list[idx])

        with open(jpg_path, 'rb') as f:
            check_chars = f.read()[-2:]

        if check_chars != b'\xff\xd9':
            print('Not complete image: ', jpg_path)
            # os.remove(jpg_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    #####################################
    # 标注数据
    #####################################
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/China/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/China_6mm/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/Europe/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/Mexico/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate/"

    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/england/"
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/england_1080p/"
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/france/"
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/italy/"
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/netherlands/"
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/moni/"
    # args.input_dir = "/yuanhuan/data/image/ZF_Europe/moni_0415/"
    args.input_dir = "/yuanhuan/data/image/ZF_Europe/hardNeg/"

    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_5M/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/sandaofangxian/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu_night_diguangzhao/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/shenzhentiaoqiao/"

    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/daminghu/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/daminghu_night/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shandongyingzikou/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shandongyingzikou_night_hongwai/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/yongzou_night_hongwai/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shenzhenlukou/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shenzhenlukou_night_hongwai/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/shenzhenlukou_night_diguangzhao/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/rongheng/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/rongheng_night_hongwai/"

    # args.input_dir = "/yuanhuan/data/image/Open_Source/MOT/MOT17/"
    # args.input_dir = "/yuanhuan/data/image/Open_Source/MOT/MOT20/"
    # args.input_dir = "/yuanhuan/data/image/Open_Source/NightOwls/nightowls/"
    # args.input_dir = "/yuanhuan/data/image/Open_Source/Cityscapes/cityscapes/"
    args.jpg_dir =  args.input_dir + "JPEGImages/"

    #####################################
    # 测试数据集
    #####################################
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/AHHBAS_418/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/AHHBAS_41a/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/AHHBAS_41c/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/AHHBAS_43c/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/AHHBAS_kakou1/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/AHHBAS_kakou2/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/AHHBAS_kakou3/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/AHHBAS_kakou4/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/AHHBAS_kakou2_night/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/AHHBAS_kakou3_night/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/AHHBPS/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/TXSDFX_c/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/TXSDFX_9/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/TXSDFX_7/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/TXSDFX_6/"
    # args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/TXSDFX_6/"

    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/banmaxian_test_image/2M_DaMingHu_far/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/banmaxian_test_image/2M_DaMingHu_near/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/banmaxian_test_image/2M_DaMingHu_night_far/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/banmaxian_test_image/2M_DaMingHu_night_near/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/banmaxian_test_image/2M_RongHeng_far/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/banmaxian_test_image/2M_RongHeng_near/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/banmaxian_test_image/2M_RongHeng_night_far/"
    # args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/banmaxian_test_image/2M_RongHeng_night_near/"
    # args.jpg_dir =  args.input_dir 

    check_jepg(args)
