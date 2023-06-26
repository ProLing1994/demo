import argparse
from auditok import split
import os
import shutil
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.utils.folder_tools import *

def data_clean(args):
    data_list = get_sub_filepaths_suffix(args.input_dir, '.wav')

    for idx in tqdm(range(len(data_list))):
        audio_path = data_list[idx]

        audio_regions = split(audio_path, 0.5, 4, 2.5, False, True)

        # check num
        num = 0
        for region in audio_regions:
            num += 1

        if num >= 2:
            audio_regions = split(audio_path, 0.5, 4, 2.5, False, True)
            idx = 0

            for region in audio_regions:
                if idx == 0:
                    output_path = audio_path.replace(
                        args.input_dir, args.output_dir)
                    create_folder(os.path.dirname(output_path))
                    filename = region.save(output_path)
                else:
                    output_path = audio_path.replace(args.input_dir, args.output_dir).split('.')[
                        0] + '_{}.wav'.format(idx)
                    create_folder(os.path.dirname(output_path))
                    filename = region.save(output_path)
                    print("Audio region saved as: {}".format(filename))

                idx += 1
        else:
            output_path = audio_path.replace(
                args.input_dir, args.output_dir)
            create_folder(os.path.dirname(output_path))
            shutil.copy(audio_path, output_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        default="/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/original/TruckIdling")
                        # default="/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/original/office")
    parser.add_argument('--output_dir', type=str,
                        default="/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/original/TruckIdling_test")
                        # default="/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/original/office_test")
    args = parser.parse_args()
    data_clean(args)


if __name__ == "__main__":
    main()
