import argparse
import os
import sys 
import shutil

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def find_xml_jpg(args):

    input_list = get_sub_filepaths_suffix(args.input_dir, suffix='.xml')
    input_list = [str(os.path.basename(input_list[idx])).split('.xml')[0] for idx in range(len(input_list))]
    print(len(input_list))

    find_input_list = get_sub_filepaths_suffix(args.find_input_dir, suffix='.jpg')
    find_input_list = [str(os.path.basename(find_input_list[idx])).split('.jpg')[0] for idx in range(len(find_input_list))]
    print(len(find_input_list))

    find_input_list = [find_input_list[idx] for idx in range(len(find_input_list)) if find_input_list[idx] in input_list]
    print(len(find_input_list))

    for idx in range(len(find_input_list)):
        input_path = os.path.join(args.find_input_dir, find_input_list[idx] + '.jpg')
        output_path = os.path.join(args.output_dir, find_input_list[idx] + '.jpg')
        print(input_path, '->', output_path)
        shutil.copy(input_path, output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_with_licenseplate/XML/"
    args.find_input_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_new/JPEGImages/"
    args.output_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_with_licenseplate/JPEGImages/"

    find_xml_jpg(args)