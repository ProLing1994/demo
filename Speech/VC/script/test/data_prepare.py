import argparse
import os
from re import L
import shutil
from tqdm import tqdm


def gen_train_data():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.spk_num = 10
    args.data_dir = "/mnt/huanyuan2/data/speech/asr/Chinese/Aishell3/train/wav/"
    args.spk_info_path = "/mnt/huanyuan2/data/speech/asr/Chinese/Aishell3/spk-info.txt"
    args.output_dir = "/mnt/huanyuan2/data/speech/vc/Chinese/vc_test/train/"

    spk_list = os.listdir(args.data_dir)
    spk_list.sort()

    # spk info
    skp_info_dict = {}
    with open(args.spk_info_path, "r") as f :
        lines = f.readlines()
        for idx in range(len(lines)):
            if idx < 3:
                continue

            spk_info = lines[idx].strip().split()
            skp_info_dict[spk_info[0]] = spk_info

    # init
    spk_male_num = 0
    spk_female_num = 0

    for idx in tqdm(range(len(spk_list))):
        spk_idx = spk_list[idx]
        spk_sex = skp_info_dict[spk_idx][2]
        
        if spk_sex == "female":

            if spk_female_num < args.spk_num:

                spk_female_num += 1 
                shutil.copytree(os.path.join(args.data_dir, spk_idx), os.path.join(args.output_dir, spk_idx))

        elif  spk_sex == "male":
            
            if spk_male_num < args.spk_num:
                
                spk_male_num += 1 
                shutil.copytree(os.path.join(args.data_dir, spk_idx), os.path.join(args.output_dir, spk_idx))


def gen_test_data():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.test_num = 10
    args.train_dir = "/mnt/huanyuan2/data/speech/vc/Chinese/vc_test/train/"
    args.test_dir = "/mnt/huanyuan2/data/speech/vc/Chinese/vc_test/test/"

    dateset_list = os.listdir(args.train_dir)
    dateset_list.sort()

    for idx in tqdm(range(len(dateset_list))):
        datasset_name = dateset_list[idx]

        # mkdir
        output_dir = os.path.join(args.test_dir, datasset_name)
        os.makedirs(output_dir)

        wav_list = os.listdir(os.path.join(args.train_dir, datasset_name))
        wav_list.sort()

        wav_list = wav_list[- args.test_num :]

        for wav_idx in range(len(wav_list)):
            wav_name = wav_list[wav_idx]
            
            shutil.move(os.path.join(args.train_dir, datasset_name, wav_name), os.path.join(output_dir, wav_name))


if __name__ == "__main__":

    # gen_train_data()

    gen_test_data()