import argparse
import os
import shutil
from tqdm import tqdm

def copy_dataset(args):
    # mkdir 
    if not os.path.exists(args.jpg_output_dir):
        os.makedirs(args.jpg_output_dir)

    # with open(args.test_file, "r") as f:
    with open(args.val_file, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            jpg_name = line.strip() + ".jpg"
            jpg_path = os.path.join(args.jpg_dir, jpg_name)
            output_jpg_path = os.path.join(args.jpg_output_dir, jpg_name)
            print(jpg_path)
            print(output_jpg_path)
            shutil.copy(jpg_path, output_jpg_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/Open_Source/Wider_Face/training_landmark/"

    args.trainval_file = args.input_dir + "ImageSets/Main/trainval.txt"
    args.train_file = args.input_dir + "ImageSets/Main/train.txt"
    args.val_file = args.input_dir + "ImageSets/Main/val.txt"
    args.test_file = args.input_dir + "ImageSets/Main/test.txt"

    args.jpg_dir = os.path.join(args.input_dir, "JPEGImages")
    # args.jpg_output_dir = os.path.join(args.input_dir, "JPEGImages_test")
    args.jpg_output_dir = os.path.join(args.input_dir, "JPEGImages_val")

    copy_dataset(args)
