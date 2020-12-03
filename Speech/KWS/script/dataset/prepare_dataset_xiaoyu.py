import argparse
import glob
import os
import shutil
import re


def get_hash_name(file_name):
    # xiaoyu spk
    hash_name = file_name.split('_')[0]
    return hash_name

def dataset_generator(directory, lables, names):
    input_dir = os.path.join(directory, "小鱼在家-数据交付2020-10-29")
    assert os.path.exists(input_dir), "[ERROR:] 不存在数据集‘小鱼在家-数据交付2020-10-29’，请检查！"

    output_dir = os.path.join(directory, "XiaoYuDataset_10292020")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # init input
    label_list = lables.split(',')
    name_list = names.split(',')
    name_list.append('random')
    
    # mkdir 
    for name in name_list:
        if not os.path.exists(os.path.join(output_dir, name)):
            os.makedirs(os.path.join(output_dir, name))

    # find wavs
    suffix = '.wav'
    image_list = glob.glob(os.path.join(input_dir, '*/*/*/' + '*' + suffix))
    for image_path in image_list:

        dir_name = os.path.basename(os.path.dirname(image_path))
        base_name = os.path.basename(image_path)

        # man/woman
        if '女' in dir_name:
            image_name = dir_name.split('_')[0] + 'M1_'
        elif '男' in dir_name:
            image_name = dir_name.split('_')[0] + 'M0_'
        else:
            raise Exception("[ERROR:] 数据文件夹中，男/女性别存在问题，请检查！")
        
        image_name += base_name

        # split for labels
        find_lable_bool = False
        for idx in range(len(label_list)):
            if label_list[idx] in base_name:
                output_path = os.path.join(output_dir, name_list[idx], image_name)
                find_lable_bool = True
                # print(image_path, ' -> ', output_path)

        if find_lable_bool == True:
            shutil.copy(image_path, output_path)
        else:
            output_path = os.path.join(output_dir, name_list[-1], image_name)
            # print(image_path, ' -> ', output_path)
            shutil.copy(image_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare XiaoYu Dataset")
    parser.add_argument('--dir', type=str, default='E:\\project\\data\\speech\\kws\\xiaoyu_dataset_03022018')
    parser.add_argument('--labels', type=str, default='小鱼小鱼,小雅小雅,小度小度,小爱同学,天猫精灵,若琪')
    parser.add_argument('--names', type=str, default='xiaoyu,xiaoya,xiaodu,xiaoaitongxue,tianmaojingling,ruoqi')
    args = parser.parse_args()
    dataset_generator(args.dir, args.labels, args.names)

if __name__ == "__main__":
    main()