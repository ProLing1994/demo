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
        # else:
        #     image = cv2.imread(jpg_path, 1)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.input_dir = "/yuanhuan/data/image/LicensePlate/China/"
    args.input_dir = "/yuanhuan/data/image/LicensePlate/China_6mm/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/Europe/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate/Mexico/"
    # args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone/"

    args.jpg_dir =  args.input_dir + "JPEGImages/"

    check_jepg(args)
