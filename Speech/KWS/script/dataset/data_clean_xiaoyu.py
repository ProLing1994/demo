import argparse
from auditok import split
import os

def data_clean(input_dir, output_dir):
    file_list = os.listdir(input_dir)

    for file_path in file_list:
        if not file_path.endswith(".wav"):
            continue
        audio_path = os.path.join(input_dir, file_path)
        audio_regions = split(audio_path, 0.5, 4, 1.5, True, True)
        idx = 0
        for region in audio_regions:
            if idx == 0:
                filename = region.save(os.path.join(output_dir, os.path.basename(audio_path)))
            else:
                filename = region.save(os.path.join(output_dir, os.path.basename(audio_path).split('.')[0] + '_{}.wav'.format(idx)))
            print("Audio region saved as: {}".format(filename))
            idx += 1

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dir', type=str, default="E:\\project\\data\\speech\\kws\\xiaoyu_dataset_03022018\\XiaoYuDataset_10292020\\xiaoyu")
    # parser.add_argument('--output_dir', type=str, default="E:\\project\\data\\speech\\kws\\xiaoyu_dataset_03022018\\test")
    parser.add_argument('--input_dir', type=str, default="E:\\project\\data\\speech\\kws\\xiaorui\\11302020\\0000000000000000-201130-141128-141431-000001001220")
    parser.add_argument('--output_dir', type=str, default="E:\\project\\data\\speech\\kws\\xiaorui\\11302020\\test")
    args = parser.parse_args()
    data_clean(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()