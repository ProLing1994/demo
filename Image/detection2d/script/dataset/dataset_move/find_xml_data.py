import argparse
import os
import sys 
import shutil

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def find_xml_data(args):

    input_list = get_sub_filepaths_suffix(args.jpg_refer_dir, suffix='.jpg')
    input_list = [str(os.path.basename(input_list[idx])).split('.jpg')[0] for idx in range(len(input_list))]
    print(len(input_list))

    find_input_list = get_sub_filepaths_suffix(args.xml_from_dir, suffix='.xml')
    find_input_list = [str(os.path.basename(find_input_list[idx])).split('.xml')[0] for idx in range(len(find_input_list))]
    print(len(find_input_list))

    find_input_list = [find_input_list[idx] for idx in range(len(find_input_list)) if find_input_list[idx] in input_list]
    print(len(find_input_list))

    for idx in range(len(find_input_list)):
        input_path = os.path.join(args.xml_from_dir, find_input_list[idx] + '.xml')
        output_path = os.path.join(args.xml_to_dir, find_input_list[idx] + '.xml')
        print(input_path, '->', output_path)
        shutil.copy(input_path, output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/RM_C28_detection/america/"

    args.jpg_refer_dir = os.path.join(args.input_dir, "JPEGImages_test")
    args.xml_from_dir = os.path.join(args.input_dir, "Annotations")
    args.xml_to_dir = os.path.join(args.input_dir, "JPEGImages_test")

    find_xml_data(args)