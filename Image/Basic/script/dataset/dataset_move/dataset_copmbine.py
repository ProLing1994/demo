import argparse
import os
import shutil
from tqdm import tqdm


def combine_dataset(args):

    # combine_file
    for idx in tqdm(range(len(args.combine_file_list))):
        combine_idx = args.combine_file_list[idx]
        tqdm.write(combine_idx)

        input_file = os.path.join(args.input_dir, combine_idx)
        output_file = os.path.join(args.output_dir, combine_idx)

        with open(input_file, "r") as f:
            input_lines = f.readlines()

        with open(output_file, "a") as f:
            for line in input_lines:
                f.write(line)

    # combine_dir
    for idx in tqdm(range(len(args.combine_dir_list))):
        combine_idx = args.combine_dir_list[idx]
        tqdm.write(combine_idx)

        input_dir = os.path.join(args.input_dir, combine_idx)
        output_dir = os.path.join(args.output_dir, combine_idx)

        input_list = os.listdir(input_dir)        
        for idy in tqdm(range(len(input_list))):
            input_path = os.path.join(input_dir, input_list[idy])
            output_path = os.path.join(output_dir, input_list[idy])

            if not os.path.exists(output_path):
                # print(input_path, '->', output_path)
                shutil.move(input_path, output_path)

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/new/rongheng_night_hongwai/"
    args.output_dir = "/yuanhuan/data/image/ZG_BMX_detection/rongheng_night_hongwai/"

    args.combine_file_list = [
                                "ImageSets/Main/trainval.txt", 
                                "ImageSets/Main/train.txt", 
                                "ImageSets/Main/val.txt", 
                                "ImageSets/Main/test.txt",
                            ]
    args.combine_dir_list = [
                                "Annotations_CarBusTruckBicyclistMotorcyclistPerson",
                                "Annotations_CarBusTruckBicyclistMotorcyclistPerson_filter",
                                "Annotations_HeadHelmet",
                                "Annotations_HeadHelmet_w_size",
                                "JPEGImages",
                                "JPEGImages_test",
                                "XML",
                            ]
    
    combine_dataset(args)