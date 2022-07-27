import argparse
import numpy as np
import os
from tqdm import tqdm


def gen_txt(args):
    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list.sort()

    with open(args.output_path, 'w') as f:
        for idx in tqdm(range(len(jpg_list))):
            f.writelines("{}\n".format(jpg_list[idx].replace('.jpg', '')))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.jpg_dir = "/yuanhuan/data/image/Distance_detection/"
    # args.jpg_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_image/SZTQ/"
    # args.jpg_dir = "/yuanhuan/data/image/ZG_BMX_detection/banmaxian_test_image/2M_RongHeng_night_near/"
    args.jpg_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu_night_diguangzhao/JPEGImages_test/"
    args.output_path = os.path.join(args.jpg_dir, "images.txt")

    gen_txt(args)
