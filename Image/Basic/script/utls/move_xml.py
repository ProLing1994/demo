import argparse
import os
import shutil
from tqdm import tqdm


def move_xml(args):

    for idx in range(len(args.data_list)):
        data_name_idx = args.data_list[idx]
        print('Date name = {}'.format(data_name_idx))

        if args.from_dataset_bool:
            args.jpg_dir = os.path.join(args.data_dir, data_name_idx, "JPEGImages/")
            args.anno_dir = os.path.join(args.data_dir, data_name_idx, args.anno_name)                          

            anno_path = os.path.join(args.jpg_dir, '%s.xml')
            anno_out_path = os.path.join(args.anno_dir, '%s.xml')

        else:
            args.jpg_dir = os.path.join(args.data_dir, data_name_idx)
            args.anno_dir = os.path.join(args.data_dir, data_name_idx + '_' + args.anno_name)

            anno_path = os.path.join(args.jpg_dir, '%s.xml')
            anno_out_path = os.path.join(args.anno_dir, '%s.xml')

        file_list = os.listdir(args.jpg_dir)
        file_list = list(set([str(os.path.basename(file)).replace('.xml', '') for file in file_list if file[-4:] == '.xml']))

        for idx in tqdm(range(len(file_list))):
            print(anno_path % (file_list[idx]))
            shutil.copy(anno_path % (file_list[idx]), anno_out_path % (file_list[idx]))
            os.remove(anno_path % (file_list[idx]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ######################################
    # 收集测试图像：
    ######################################

    args.data_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/"

    # args.data_list = ['jiayouzhan_test_image/2MB', 'jiayouzhan_test_image/2MH' ]
    # args.data_list = ['jiayouzhan_test_image/5MB', 'jiayouzhan_test_image/5MH' ]
    # args.data_list = ['jiayouzhan_test_image/SZTQ' ]
    # args.data_list = ['jiayouzhan_test_image/SDFX_B1', 'jiayouzhan_test_image/SDFX_B2', 'jiayouzhan_test_image/SDFX_H1', 'jiayouzhan_test_image/SDFX_H2' ]
    # args.data_list = ['jiayouzhan_test_image/AHHBAS_41a', 'jiayouzhan_test_image/AHHBAS_41c', 'jiayouzhan_test_image/AHHBAS_43c', 'jiayouzhan_test_image/AHHBAS_418' ]
    # args.data_list = ['jiayouzhan_test_image/AHHBAS_kakou1', 'jiayouzhan_test_image/AHHBAS_kakou2', 'jiayouzhan_test_image/AHHBAS_kakou3', 'jiayouzhan_test_image/AHHBAS_kakou4' ]
    # args.data_list = ['jiayouzhan_test_image/TXSDFX_6', 'jiayouzhan_test_image/TXSDFX_7', 'jiayouzhan_test_image/TXSDFX_9', 'jiayouzhan_test_image/TXSDFX_c' ]
    # args.data_list = ['jiayouzhan_test_image/AHHBPS' ]
    args.data_list = ['jiayouzhan_test_image/2MB', 'jiayouzhan_test_image/2MH', 'jiayouzhan_test_image/5MB', 'jiayouzhan_test_image/5MH',
                      'jiayouzhan_test_image/SZTQ', 
                      'jiayouzhan_test_image/SDFX_B1', 'jiayouzhan_test_image/SDFX_B2', 'jiayouzhan_test_image/SDFX_H1', 'jiayouzhan_test_image/SDFX_H2',
                      'jiayouzhan_test_image/AHHBAS_41a', 'jiayouzhan_test_image/AHHBAS_41c', 'jiayouzhan_test_image/AHHBAS_43c', 'jiayouzhan_test_image/AHHBAS_418', 
                      'jiayouzhan_test_image/AHHBAS_kakou1', 'jiayouzhan_test_image/AHHBAS_kakou2', 'jiayouzhan_test_image/AHHBAS_kakou3', 'jiayouzhan_test_image/AHHBAS_kakou4',
                      'jiayouzhan_test_image/TXSDFX_6', 'jiayouzhan_test_image/TXSDFX_7', 'jiayouzhan_test_image/TXSDFX_9', 'jiayouzhan_test_image/TXSDFX_c', 
                      'jiayouzhan_test_image/AHHBPS' ]

    args.from_dataset_bool = False
    
    args.anno_name = 'XML'
    
    move_xml(args)