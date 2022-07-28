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
            jpg_path = os.path.join(args.jpg_dir, line.strip() + '.jpg')
            output_jpg_path = os.path.join(args.jpg_output_dir, line.strip() + '.jpg')
            shutil.copy(jpg_path, output_jpg_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/ZG_ZHJYZ_detection/anhuihuaibeigaosu_night_diguangzhao/"

    args.trainval_file = args.input_dir + "ImageSets/Main/trainval.txt"
    args.train_file = args.input_dir + "ImageSets/Main/train.txt"
    args.val_file = args.input_dir + "ImageSets/Main/val.txt"
    args.test_file = args.input_dir + "ImageSets/Main/test.txt"

    args.jpg_dir =  args.input_dir + "JPEGImages/"
    args.jpg_output_dir =  args.input_dir + "JPEGImages_test/"

    copy_dataset(args)
