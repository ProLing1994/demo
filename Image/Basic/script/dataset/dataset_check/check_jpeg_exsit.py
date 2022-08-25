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

    # 查找数据集
    find_list = []
    for idx in tqdm(range(len(jpg_list))):
        jpg_idx = jpg_list[idx]

        if jpg_idx in exist_jpg_list:
            find_list.append(jpg_idx)
    
    # 删除已有数据集
    print("Find jpg num:", len(find_list))
    for idx in tqdm(range(len(find_list))):
        jpg_idx = find_list[idx]
        jpg_path = os.path.join(args.jpg_dir, jpg_idx)
        print(jpg_path)
        # os.remove(jpg_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.jpg_dir = "/yuanhuan/data/image/ZG_BMX_detection/new/rongheng_night_hongwai/JPEGImages/"
    args.check_dataset_list = [ "/yuanhuan/data/image/ZG_BMX_detection/rongheng_night_hongwai/JPEGImages/" ]

    check_jepg_exist(args)