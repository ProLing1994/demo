import argparse
import glob
import os
import shutil
import re


def get_hash_name(file_name):
    # xiaoyu spk
    hash_name = file_name.split('_')[0]

    if 'XIAORUI' in file_name:
        hash_name = file_name.split('_')[4][:4]
    # print(hash_name)
    return hash_name


def dataset_generator(input_dir, output_dir, lables, names):
    assert os.path.exists(input_dir), "[ERROR:] 不存在数据集，请检查！"

    # init input
    label_list = lables.split(',')
    label_list.append('random')
    name_list = names.split(',')
    name_list.append('random')
    
    # mkdir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for name in name_list:
        if not os.path.exists(os.path.join(output_dir, name)):
            os.makedirs(os.path.join(output_dir, name))

    # find wavs
    suffix = '.wav'
    image_list = glob.glob(os.path.join(input_dir, '*/*/' + '*' + suffix))
    for image_path in image_list:

        dir_name = os.path.basename(os.path.dirname(image_path))
        base_name = os.path.basename(image_path)

        # split for labels
        find_lable_bool = False
        for idx in range(len(label_list)):
            if label_list[idx] in base_name:
                output_path = os.path.join(output_dir, name_list[idx], base_name)
                find_lable_bool = True

        if find_lable_bool == True:
            print(image_path, ' -> ', output_path)
            # shutil.copy(image_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare XiaoYu Dataset")
    parser.add_argument('--input_dir', type=str, default='E:\\project\\data\\speech\\kws\\xiaorui\\12022020')
    parser.add_argument('--output_dir', type=str, default='E:\\project\\data\\speech\\kws\\xiaorui\\XiaoRuiDataset_12022020')
    parser.add_argument('--labels', type=str, default='xiaorui')
    parser.add_argument('--names', type=str, default='xiaorui')
    args = parser.parse_args()
    dataset_generator(args.input_dir, args.output_dir, args.labels, args.names)

if __name__ == "__main__":
    main()