import argparse
import numpy as np
import os
from tqdm import tqdm


def check_jepg_exist(args):
    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]

    # 收集已有数据集
    exist_jpg_list = []
    for idx in range(len(args.check_dataset_list)):
        dataset_idx = args.check_dataset_list[idx]
        jpg_idx_list = np.array(os.listdir(dataset_idx))
        jpg_idx_list = jpg_idx_list[[jpg.endswith('.jpg') for jpg in jpg_idx_list]]
        exist_jpg_list.extend(jpg_idx_list)

    for idx in tqdm(range(len(jpg_list))):
        jpg_idx = jpg_list[idx]

        assert jpg_idx not in exist_jpg_list


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_new/"
    args.jpg_dir =  args.input_dir + "JPEGImages/"
    args.check_dataset_list = [ "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone/" + "JPEGImages/" ]

    check_jepg_exist(args)