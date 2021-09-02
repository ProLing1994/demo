import argparse
import os
import pandas as pd
import shutil
import sys
import random

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.folder_tools import *

def add_dataset(args):
    '''
    目的：验证 TTS 合成数据的可行性，能够提升 KWS 算法性能
    方法：在 TTS 数据上，按照不同百分比添加重庆录制数据，使用 TTS + 重庆录制数据训练模型，测试模型在重庆录制数据/深圳录制数据上的召回率
    数据来源：
        TTS：8072/6879/1193(总数/训练/测试)
        重庆：3304/2671/633(总数/训练/测试)，说话人：22/17/5(总数/训练/测试)
        深圳：2706/2052/654(总数/训练/测试)，说话人：53/41/12(总数/训练/测试)
    '''
    # init
    random.seed(0)
    input_dir = args.tts_folder
    create_folder(args.output_tts_folder)
    create_folder(args.output_add_folder)
    shutil.copy(os.path.join(input_dir, "background_noise_files.csv"), args.output_tts_folder) 
    shutil.copy(os.path.join(input_dir, "background_noise_files.csv"), args.output_add_folder) 

    # load csv
    data_pd = pd.read_csv(os.path.join(input_dir, "total_data_files.csv"))
    add_data_pd = pd.read_csv(os.path.join(args.add_folder, "total_data_files.csv"))

    # 添加 label_name, speaker_id, select_id 
    label_name_list = []
    speaker_id_list = []
    for _, row in add_data_pd.iterrows():
        file_name = os.path.basename(row['file'])
        label_name = os.path.basename(row['label'])
        mode_name = os.path.basename(row['mode'])
        speaker_id = int(-1)
        if "RM_KWS_ACTIVATEBWC_activatebwc_" in file_name and mode_name == "training":
            speaker_id = int(str(file_name).split("_")[-1][1:4]) 
        if label_name == "activatebwc":
            label_name = "tts"
        speaker_id_list.append(speaker_id)
        label_name_list.append(label_name)

    speaker_id_set = list(set(speaker_id_list))
    speaker_id_set.remove(-1)
    add_speaker_id_num = int(args.add_frequency * len(speaker_id_set))
    speaker_id_select_list = random.sample(speaker_id_set, add_speaker_id_num)
    select_id_list = [True if speaker_id_list[idx] in speaker_id_select_list else False for idx in range(len(speaker_id_list))]
    print("speaker_num: {}/{}, {}".format(len(speaker_id_select_list), len(speaker_id_set), speaker_id_select_list))

    add_data_pd["label"] = label_name_list
    add_data_pd["speaker_id"] = speaker_id_list
    add_data_pd["select_id"] = select_id_list
    
    # 数据划分，合并数据
    merge_data_pd = add_data_pd[add_data_pd['label'] == "tts"]
    merge_data_pd = merge_data_pd[merge_data_pd['mode'] == "training"]
    merge_data_pd = merge_data_pd[merge_data_pd['select_id'] == True]
    merge_data_pd = pd.concat([data_pd, merge_data_pd])
    merge_data_pd.to_csv(os.path.join(args.output_tts_folder, "total_data_files.csv"), index=False, encoding="utf_8_sig")
    
    # 生成对比数据，用于对比实验结果
    drop_list = []
    for idx, row in add_data_pd.iterrows():
        if row['label'] == "tts" and row['mode'] == "training" and row['select_id'] == False:
            drop_list.append(idx)

    drop_data_pd = add_data_pd.drop(drop_list) 
    drop_data_pd.to_csv(os.path.join(args.output_add_folder, "total_data_files.csv"), index=False, encoding="utf_8_sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Streamax KWS Data Split Engine')
    args = parser.parse_args()
    args.tts_folder = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_tts_2s_1.8_08272021/"
    args.add_folder = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.5_03312021/"
    args.output_tts_folder = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_tts_2s_1.9.10_08302021/"
    args.output_add_folder = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.5.10_03312021/"
    args.add_frequency = 0.1
    add_dataset(args)