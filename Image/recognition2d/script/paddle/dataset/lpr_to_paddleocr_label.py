import argparse
import os
import shutil
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
# sys.path.insert(0, '/yuanhuan/demo')
from Image.Basic.utils.folder_tools import *


def lpr_to_paddleocr_label(args):

    for format_idx in range(len(args.format_list)):
        format_path = args.format_list[format_idx]
        
        # input list 
        input_list = []
        input_path = os.path.join(args.input_dir, format_path)
        print(args.input_dir)
        print(format_path)
        print(input_path)
        with open(input_path, "r") as f:
            for line in f:
                input_list.append(line.strip())
        
        # output path
        output_label_txt = os.path.join(args.output_dir, format_path)

        # mkdir
        create_folder(os.path.dirname(output_label_txt))

        # run
        with open(output_label_txt, "w") as f:
            for idx in tqdm(range(len(input_list))):

                input_path = input_list[idx]
                input_name = os.path.basename(input_path)
                input_label = str(input_name).split('.jpg')[0].split('_')[-1]

                f.write('{}'.format(input_path))
                f.write("\t")
                f.write('{}'.format(input_label))
                f.write("\n")
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_202301/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/model/image/lpr/paddle_dict/plate_zd_mask_202301/") 
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, "ImageSets")

    args.format_list = [
                        "ImageSets/Main/trainval.txt", 
                        "ImageSets/Main/train.txt",
                        "ImageSets/Main/val.txt",
                        "ImageSets/Main/test.txt",
                        ]

    lpr_to_paddleocr_label(args)
    