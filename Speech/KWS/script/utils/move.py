import argparse
import os
import shutil
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/audio/客户采集音频/20230612/"
    args.output_dir = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/office/"

    file_list = get_sub_filepaths_suffix(args.input_dir, '.wav')

    for file_idx in tqdm(range(len(file_list))):
        file = file_list[file_idx]

        person_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file))))

        img_path = file
        out_img_path = img_path.replace(args.input_dir, args.output_dir).replace(person_name, '')
        # print(img_path, out_img_path)
        create_folder(os.path.dirname(out_img_path))
        assert not os.path.exists(out_img_path)
        # shutil.copy(img_path, out_img_path)
        shutil.move(img_path, out_img_path)