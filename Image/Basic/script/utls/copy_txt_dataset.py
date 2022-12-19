import argparse
import os
import shutil
from tqdm import tqdm

def copy_dataset(args):
    # mkdir 
    if not os.path.exists(args.jpg_output_dir):
        os.makedirs(args.jpg_output_dir)

    with open(args.test_file, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            # normal
            # jpg_path = os.path.join(args.jpg_dir, line.strip() + '.jpg')
            # output_jpg_path = os.path.join(args.jpg_output_dir, line.strip() + '.jpg')
            # shutil.copy(jpg_path, output_jpg_path)
            
            # # seg plate
            # jpg_name = os.path.basename(line.strip().split(' ')[0])
            # jpg_path = os.path.join(args.jpg_dir, jpg_name)
            # output_jpg_path = os.path.join(args.jpg_output_dir, jpg_name)
            # shutil.copy(jpg_path, output_jpg_path)

            jpg_name = os.path.basename(line.strip().split(' ')[0])
            jpg_path = line.strip().split(' ')[0]
            output_jpg_path = os.path.join(args.jpg_output_dir, jpg_name)
            shutil.copy(jpg_path, output_jpg_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask/"
    args.test_file = args.input_dir + "ocr_merge_test/ImageSets/Main/test.txt"

    args.jpg_dir =  args.input_dir + "Images/"
    args.jpg_output_dir =  args.input_dir + "Images_ocr_merge_test/"

    copy_dataset(args)
