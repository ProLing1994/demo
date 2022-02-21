import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm


def getImageVar(image):
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return imageVar


def check_clarity(args):
    # mkdir 
    if not os.path.exists( args.output_dir ):
        os.makedirs( args.output_dir )

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    
    for idx in tqdm(range(len(jpg_list))):
        jpg_apth = os.path.join(args.jpg_dir, jpg_list[idx])

        image = cv2.imread(jpg_apth)
        imageVar = getImageVar(image)
        print("{}: {:.2f}".format(jpg_apth, imageVar))

        output_img_path = os.path.join(args.output_dir, jpg_list[idx].replace(".jpg", "_{:.2f}.jpg".format(imageVar)))
        cv2.imwrite(output_img_path, image)

        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.input_dir = "/mnt/huanyuan2/data/image/LicensePlate/Mexico/"
    # args.jpg_dir =  args.input_dir + "LicensePlate_crop/"
    # args.output_dir = args.input_dir + "LicensePlate_crop_clarity/"

    args.input_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/"
    args.jpg_dir =  args.input_dir + "LicensePlate_crop/"
    args.output_dir = args.input_dir + "LicensePlate_crop_clarity/"

    check_clarity(args)
