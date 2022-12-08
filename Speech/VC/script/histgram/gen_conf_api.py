import argparse
import os
from tqdm import tqdm


def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--data_dir', type=str,  default="/mnt/huanyuan2/data/speech/asr/English/VCC2020-database/dataset/train/", help='data dir')
    # parser.add_argument('-o', '--output_dir', type=str,  default="/mnt/huanyuan/data/speech/vc/English_dataset/dataset_audio_hdf5/VCC2020/conf_test/", help='output dir')
    parser.add_argument('-d', '--data_dir', type=str,  default="/mnt/huanyuan2/data/speech/vc/Chinese/vc_test/train/", help='data dir')
    parser.add_argument('-o', '--output_dir', type=str,  default="/mnt/huanyuan/data/speech/vc/Chinese_dataset/dataset_audio_hdf5/BZNSYP_Aishell3/conf_test/", help='output dir')
    args = parser.parse_args()

    spk_list = os.listdir(args.data_dir)
    spk_list.sort()

    # mkdir 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for spk_idx in tqdm(range(len(spk_list))):
        spk_name = spk_list[spk_idx]

        f0_path = os.path.join(args.output_dir, spk_name + '.f0')
        min_f0 = int(input("请输入 spk: {}, min_f0: ".format(spk_name)))
        max_f0 = int(input("请输入 spk: {}, max_f0: ".format(spk_name)))
        with open(f0_path, 'w') as file:
            file.write("{} {}\n".format(min_f0, max_f0))

        npow_path = os.path.join(args.output_dir, spk_name + '.pow')
        npow = float(input("请输入 spk: {}, npow: ".format(spk_name)))
        with open(npow_path, 'w') as file:
            file.write("{}\n".format(npow))


if __name__ == '__main__':
    main()