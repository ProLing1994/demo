import argparse
import cv2
from importlib.resources import path
import numpy as np
import os
from tqdm import tqdm


def bmp_to_jpg(args):
    img_list = np.array(os.listdir(args.jpg_dir))
    img_list.sort()

    for idx in tqdm(range(len(img_list))):

        img_path = os.path.join(args.jpg_dir, img_list[idx])

        if img_path.endswith('.bmp'):

            img = cv2.imread(img_path, -1)
            cv2.imwrite(img_path.replace('.bmp','.jpg'), img)
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.jpg_dir = "/yuanhuan/data/image/Distance_detection/"

    bmp_to_jpg(args)
