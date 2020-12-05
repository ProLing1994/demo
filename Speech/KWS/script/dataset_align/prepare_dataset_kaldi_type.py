import argparse
import os
import sys

from tqdm import tqdm


def main():
    defaullt_dataset_path = "/mnt/huanyuan/data/speech/kws/xiaorui_dataset/original_dataset/XiaoRuiDataset_12022020/"
    default_dest_path = "/mnt/huanyuan/data/speech/kws/xiaorui_dataset/original_dataset/XiaoRuiDataset_12022020/kaldi_type"
    default_keyword_list = "xiaorui"
    default_keyword_chinese_name_list = "小 锐 小 锐"

    parser = argparse.ArgumentParser(description = "Data preparation");
    parser.add_argument('--dataset_path', type=str,default=defaullt_dataset_path,dest='dataset_dir',help='Raw dataset path')
    parser.add_argument('--dest_path',type=str,default=default_dest_path,dest='dest_dir',help='Data preparation in Kaldi type')
    parser.add_argument('--keyword_list',type=str,default=default_keyword_list)
    parser.add_argument('--keyword_chinese_name_list',type=str,default=default_keyword_chinese_name_list)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    dest_dir = args.dest_dir
    keyword_list = args.keyword_list.split(",")
    keyword_chinese_name_list = args.keyword_chinese_name_list.split(",")

    os.system("mkdir -p "+ dest_dir)
    f1 = open(dest_dir + "/text","w")
    f2 = open(dest_dir + "/wav.scp","w")
    f3 = open(dest_dir + "/utt2spk","w")

    folder_list = os.listdir(dataset_dir)
    for keyword_idx in range(len(keyword_list)):
        keyword = keyword_list[keyword_idx]
        keyword_chinese_name = keyword_chinese_name_list[keyword_idx]
        if not keyword in folder_list:
            continue

        keyword_dir = os.path.join(dataset_dir, keyword)
        file_list = os.listdir(keyword_dir)
        file_list.sort()

        for file_name in tqdm(file_list):
            file_path = os.path.join(keyword_dir, file_name)
            
            if "唤醒词_小鱼小鱼" in file_name:
                spk = file_name.strip().split('.')[0].split('_')[0]
                device = 'D0'
                text = file_name.strip().split('_')[-1].split('.')[0]
            elif "RM_KWS_XIAOYU_" in file_name:
                spk = file_name.strip().split('_')[-1].split('.')[0][:6]
                device = file_name.strip().split('_')[-1].split('.')[0][6:8]
                text = file_name.strip().split('_')[-1].split('.')[0][8:]
            elif "RM_KWS_XIAORUI_" in file_name:
                spk = file_name.strip().split('_')[-1].split('.')[0][:6]
                device = file_name.strip().split('_')[-1].split('.')[0][6:8]
                text = file_name.strip().split('_')[-1].split('.')[0][8:]
            else:
                raise Exception('[ERROR] Unknow file_name, please check!')
            
            f1.writelines(spk+"-"+keyword+"-"+device+"-"+text + " " + keyword_chinese_name + "\n")
            f2.writelines(spk+"-"+keyword+"-"+device+"-"+text + " " + file_path+"\n")
            f3.writelines(spk+"-"+keyword+"-"+device+"-"+text + " " + spk+"\n" )
    f1.close()
    f2.close()
    f3.close()

if __name__ == "__main__":
    main()