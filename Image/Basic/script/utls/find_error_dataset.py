import argparse
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def find_dataaset(args):

    for idx in range(len(args.data_list)):
        data_name_idx = args.data_list[idx]
        print('Date name = {}'.format(data_name_idx))

        if args.from_dataset_bool:
            args.jpg_dir = os.path.join(args.data_dir, data_name_idx, "JPEGImages/")
            args.anno_dir = os.path.join(args.data_dir, data_name_idx, args.anno_name)                 
            args.jpg_output_dir = os.path.join(args.output_dir, data_name_idx, "JPEGImages/")
            args.anno_output_dir = os.path.join(args.output_dir, data_name_idx, "JPEGImages/")            
            args.cache_output_dir = os.path.join(args.output_dir, data_name_idx, "cache/")            

            jpg_path = os.path.join(args.jpg_dir, '%s.jpg')
            anno_path = os.path.join(args.anno_dir, '%s.xml')
            jpg_out_path = os.path.join(args.jpg_output_dir, '%s.jpg')
            anno_out_path = os.path.join(args.anno_output_dir, '%s.xml')
            cache_dir = os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')) + '_' + test_name, 'results', args.result_folder)
            cache_out_dir = os.path.join(args.cache_output_dir, test_name)
            
            create_folder(args.anno_output_dir)
            create_folder(args.jpg_output_dir)

        else:
            args.jpg_dir = os.path.join(args.data_dir, data_name_idx)
            args.anno_dir = os.path.join(args.data_dir, data_name_idx + '_' + args.anno_name)
            args.jpg_output_dir = os.path.join(args.output_dir, data_name_idx)
            args.anno_output_dir = os.path.join(args.output_dir, data_name_idx)      
            args.cache_output_dir = os.path.join(args.output_dir, data_name_idx + "_cache/")       
            
            jpg_path = os.path.join(args.jpg_dir, '%s.jpg')
            anno_path = os.path.join(args.anno_dir, '%s.xml')
            jpg_out_path = os.path.join(args.jpg_output_dir, '%s.jpg')
            anno_out_path = os.path.join(args.anno_output_dir, '%s.xml')
            cache_dir = os.path.join(args.model_dir, '_'.join(str(data_name_idx).split('/')), 'results', args.result_folder)
            cache_out_dir = os.path.join(args.cache_output_dir, test_name)

            create_folder(args.anno_output_dir)
            create_folder(args.jpg_output_dir)

        shutil.copytree(cache_dir, cache_out_dir)
        file_list = get_sub_filepaths_suffix(cache_dir, suffix='.jpg')
        file_list = list(set([str(os.path.basename(file)).replace('.jpg', '') for file in file_list]))

        for idx in tqdm(range(len(file_list))):
            shutil.copy(jpg_path % (file_list[idx]), jpg_out_path % (file_list[idx]))
            shutil.copy(anno_path % (file_list[idx]), anno_out_path % (file_list[idx]))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    #####################################
    # Car_Bus_Truck_Licenseplate
    # 测试集图像
    #####################################
    args.data_dir = "/yuanhuan/data/image/"
    args.output_dir = "/yuanhuan/data/image/ZG_Refine_dateset/"

    # args.data_list = ['ZG_ZHJYZ_detection/jiayouzhan', 'ZG_ZHJYZ_detection/jiayouzhan_5M', 'ZG_ZHJYZ_detection/sandaofangxian', 'ZG_AHHBGS_detection/anhuihuaibeigaosu']
    args.data_list = ['ZG_ZHJYZ_detection/shenzhentiaoqiao']

    args.from_dataset_bool = True

    # ######################################
    # # 收集测试图像：
    # ######################################

    # args.data_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/"
    # args.output_dir = "/yuanhuan/data/image/ZG_Refine_dateset/"

    # # args.data_list = ['jiayouzhan_test_image/2MB', 'jiayouzhan_test_image/2MH' ]
    # # args.data_list = ['jiayouzhan_test_image/5MB', 'jiayouzhan_test_image/5MH' ]
    # # args.data_list = ['jiayouzhan_test_image/SZTQ' ]
    # # args.data_list = ['jiayouzhan_test_image/SDFX_B1', 'jiayouzhan_test_image/SDFX_B2', 'jiayouzhan_test_image/SDFX_H1', 'jiayouzhan_test_image/SDFX_H2' ]
    # # args.data_list = ['jiayouzhan_test_image/AHHBAS_41a', 'jiayouzhan_test_image/AHHBAS_41c', 'jiayouzhan_test_image/AHHBAS_43c', 'jiayouzhan_test_image/AHHBAS_418' ]
    # # args.data_list = ['jiayouzhan_test_image/AHHBAS_kakou1', 'jiayouzhan_test_image/AHHBAS_kakou2', 'jiayouzhan_test_image/AHHBAS_kakou3', 'jiayouzhan_test_image/AHHBAS_kakou4' ]
    # # args.data_list = ['jiayouzhan_test_image/TXSDFX_6', 'jiayouzhan_test_image/TXSDFX_7', 'jiayouzhan_test_image/TXSDFX_9', 'jiayouzhan_test_image/TXSDFX_c' ]
    # # args.data_list = ['jiayouzhan_test_image/AHHBPS' ]
    # args.data_list = ['jiayouzhan_test_image/2MB', 'jiayouzhan_test_image/2MH', 'jiayouzhan_test_image/5MB', 'jiayouzhan_test_image/5MH',
    #                   'jiayouzhan_test_image/SZTQ', 
    #                   'jiayouzhan_test_image/SDFX_B1', 'jiayouzhan_test_image/SDFX_B2', 'jiayouzhan_test_image/SDFX_H1', 'jiayouzhan_test_image/SDFX_H2',
    #                   'jiayouzhan_test_image/AHHBAS_41a', 'jiayouzhan_test_image/AHHBAS_41c', 'jiayouzhan_test_image/AHHBAS_43c', 'jiayouzhan_test_image/AHHBAS_418', 
    #                   'jiayouzhan_test_image/AHHBAS_kakou1', 'jiayouzhan_test_image/AHHBAS_kakou2', 'jiayouzhan_test_image/AHHBAS_kakou3', 'jiayouzhan_test_image/AHHBAS_kakou4',
    #                   'jiayouzhan_test_image/TXSDFX_6', 'jiayouzhan_test_image/TXSDFX_7', 'jiayouzhan_test_image/TXSDFX_9', 'jiayouzhan_test_image/TXSDFX_c', 
    #                   'jiayouzhan_test_image/AHHBPS' ]

    # args.from_dataset_bool = False

    ######################################

    # SSD_VGG_FPN_RFB_2022-04-25-18_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate
    args.model_dir = "/yuanhuan/model/image/ssd_rfb/weights/SSD_VGG_FPN_RFB_2022-04-25-18_focalloss_4class_car_bus_truck_licenseplate_softmax_zg_w_fuzzy_plate/eval_epoches_299/"

    test_name = "trainval"
    # test_name = "val"
    # test_name = "test"

    args.result_folder = 'img_res_0.5_roi'

    args.anno_name = 'XML'

    find_dataaset(args)