import argparse
import glob
import os
import shutil
import re


def get_hash_name(file_name):
    if "唤醒词" in file_name:
        hash_name = file_name.strip().split('.')[0].split('_')[0]
    elif 'XIAORUI' in file_name:
        hash_name = file_name.strip().split('_')[-1].split('.')[0][:6]
    elif "小乐小乐" in file_name:
        hash_name = file_name.strip().split('-')[0].split('_')[1]
    elif "XIAOYU" in file_name:
        hash_name = file_name.strip().split('_')[-1].split('.')[0][:6]
    else:
        hash_name = file_name.strip().split('.')[0].split('_')[0]
    return hash_name


def rename(input_dir, output_dir, device_id=0):
    """
    :param device_id：          设备，单兵 0|Jabra 桌面录音设备 1
    """
    image_list = os.listdir(input_dir)
    image_list.sort()
    for image_name in image_list:
        temp = image_name.split('_')[-1].split('.')[0]
        output_name = "_".join(image_name.split('_')[:-1]) + "_S{:0>3d}M{:0>1d}D{:0>2d}T{:0>3d}".format(int(temp[1:4]), int(temp[5]), device_id * 10 + int(temp[7]), int(temp[9:])) + '.wav'
        print(image_name, '->', output_name)

        image_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, output_name)
        # os.rename(image_path, output_path)
        

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

    # dataset_generator
    # dataset_generator(args.input_dir, args.output_dir, args.labels, args.names)

    # rename
    rename("/mnt/huanyuan/data/speech/kws/xiaorui_dataset/original_dataset/XiaoRuiDataset_12082020/xiaorui/", 
            "/mnt/huanyuan/data/speech/kws/xiaorui_dataset/original_dataset/XiaoRuiDataset_12082020/xiaorui/",
            1)

if __name__ == "__main__":
    main()