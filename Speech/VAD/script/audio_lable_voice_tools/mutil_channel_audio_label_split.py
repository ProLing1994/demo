import argparse
import librosa
import os
import pandas as pd
import sys
import wave 

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/common/common')
from utils.python.folder_tools import *


def find_time_id(args, speaker_id):
    # init 
    time_id = 0

    # find_path:
    find_path_list = []
    for folder_idx in range(len(args.find_folder_list)):
        find_folder_path = args.find_folder_list[folder_idx]
        find_path_list += get_sub_filepaths_suffix(find_folder_path, ".wav")

    for equipment_idx in range(len(args.equipment_name_list)):
        equipment_id = args.equipment_id_list[equipment_idx]
        equipment_location = args.equipment_location_list[equipment_idx]

        output_format_split = args.output_format.format(speaker_id, equipment_id, equipment_location, 0).split("_")
        find_match_name = "_".join(output_format_split[:-1]) + "_" + output_format_split[-1].split("T")[0]
        temp_time_id = 0
        for find_path in find_path_list:
            if find_match_name in find_path:
                find_path_time = int(os.path.basename(find_path).split("_")[-1].split("T")[1].split(".")[0])
                if find_path_time > temp_time_id:
                    temp_time_id = find_path_time
        if temp_time_id > time_id:
            time_id = temp_time_id
    return time_id + 1

def audio_lable_split(args):
    # id_name_list
    id_name_pd = pd.read_csv(args.id_name_csv, encoding='utf_8_sig')
    name_list = id_name_pd["name"].to_list()
    id_name_list = []                               # [{'id':(), 'name':()}]
    for idx, row in id_name_pd.iterrows(): 
        id_name_list.append({'id': row['id'], 'name': row['name']}) 

    # id_name_test_list
    id_name_test_pd = pd.read_csv(args.id_name_test_csv, encoding='utf_8_sig')
    name_test_list = id_name_test_pd["name"].to_list()

    # file list 
    file_list = get_sub_filepaths(args.input_folder)
    file_list.sort()

    for idx in tqdm(range(len(file_list))):
        if not file_list[idx].endswith('.txt'):
            continue

        # id_name
        audio_name = os.path.basename(file_list[idx]).split('.')[0]
        if audio_name in name_test_list:
            continue
        
        if audio_name in name_list:
            speaker_id = id_name_pd[id_name_pd["name"] == audio_name]["id"].to_list()[0]
            time_id = find_time_id(args, speaker_id)

        else:
            speaker_id = len(id_name_list) + 1
            time_id = 0
            id_name_list.append({'id': speaker_id, 'name': audio_name}) 

        # label path
        label_path = file_list[idx]

        for segment_label_idx in range(len(args.segment_label_list)):
            segment_label = args.segment_label_list[segment_label_idx]

            for equipment_idx in range(len(args.equipment_name_list)):
                equipment_name = args.equipment_name_list[equipment_idx]
                equipment_id = args.equipment_id_list[equipment_idx]
                equipment_location = args.equipment_location_list[equipment_idx]
                expansion_rate_front = args.expansion_rate_front_list[equipment_idx]
                expansion_rate_back = args.expansion_rate_back_list[equipment_idx]
                segment_sample_rate = args.segment_sample_rate_list[0] if audio_name in args.segment_sample_rate_8k_name_list else args.segment_sample_rate_list[1]

                audio_path = label_path.split('.')[0] + '_' + equipment_name + args.audio_suffix
                if not os.path.exists(audio_path):
                    continue
                
                # audio segments
                audio_segments = []
                f = open(label_path, 'r', encoding='utf-8')
                lines = f.readlines()
                for line_idx in range(len(lines)):
                    line = lines[line_idx]
                    if line.split(':')[-1].strip() == segment_label:
                        audio_segments.append([max(0, int(line.split(':')[0].split('~')[0]) + expansion_rate_front * segment_sample_rate), int(line.split(':')[0].split('~')[1]) + expansion_rate_back * segment_sample_rate, time_id + line_idx])
                f.close()

                # output audio_segment
                audio_data = librosa.core.load(audio_path, sr=args.sample_rate)[0]
                for segment_idx in range(len(audio_segments)):
                    audio_segment = audio_segments[segment_idx]
                    audio_segment_data = audio_data[int(audio_segment[0] * (args.sample_rate / segment_sample_rate)) : int(audio_segment[1] * (args.sample_rate / segment_sample_rate))]

                    # output 
                    output_dir = os.path.join(args.output_folder, segment_label, equipment_name)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    output_path = os.path.join(output_dir, args.output_format.format(speaker_id, equipment_id, equipment_location, audio_segment[2]))
                    temp_path = os.path.join(args.output_folder, '{}{}'.format('temp', args.audio_suffix))
                    librosa.output.write_wav(temp_path, audio_segment_data, sr=args.sample_rate) 
                    os.system('sox {} -b 16 -e signed-integer {}'.format(temp_path, output_path))

    id_name_pd = pd.DataFrame(id_name_list)
    id_name_pd.to_csv(args.id_name_csv, index=False, encoding="utf_8_sig")

        
def main():
    parser = argparse.ArgumentParser(description="")
    args = parser.parse_args()
    # args.input_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_05132021/处理音频_0422/"
    # args.output_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_05132021/xiaoan_0422/"
    # args.output_format = "RM_KWS_XIAOAN_xiaoan_S{:0>3d}M0D{}{}T{:0>3d}.wav"
    # args.id_name_csv = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/唤醒词记录.csv"
    args.input_folder = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/original_dataset/ActivateBWC_07162021/activatebwc/数据处理/"
    args.output_folder = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/original_dataset/ActivateBWC_07162021/activatebwc/数据处理结果/"
    args.output_format = "RM_KWS_ACTIVATEBWC_activatebwc_S{:0>3d}M0D{}{}T{:0>3d}.wav"
    args.id_name_csv = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/original_dataset/唤醒词记录.csv"
    args.id_name_test_csv = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/original_dataset/唤醒词记录_测试人员.csv"
    args.find_folder_list = ["/mnt/huanyuan/data/speech/kws/english_kws_dataset/original_dataset/ActivateBWC_07162021/activatebwc/数据处理结果/",
                                "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/KwsEnglishDataset/activatebwc/"]

    # params
    # args.equipment_name_list = ['adpro', 'mic', 'danbin_ori', 'danbin_asr']
    # args.equipment_id_list = [4, 6, 0, 7]
    # args.equipment_location_list = [2, 1, 1, 1]
    args.equipment_name_list = ['danbin_asr', 'danbin_ori']
    args.equipment_id_list = [7, 0]
    args.equipment_location_list = [1, 1]

    # xiaoan\xiaoan1\error\error1
    # args.segment_label_list = ["xiaoan", "xiaoan1", "error", "error1"]
    # args.segment_sample_rate_list = [8000, 16000]
    # args.segment_sample_rate_8k_name_list = ["钟国胜", "梁昊", "林日丹", "李文达", "雍洪", "章钰雯", "陈彦芸", "黄凯龙", "张莹莹", "颜苑婷", "杨莹丽", "黄俊斌", "赵春海", "陈泽敏", "叶智豪", "梁嘉冠", "黄丽琼", "杨章林", "冯晓欣", "赵验", "吴玉如"]
    # args.sample_rate = 16000
    # args.expansion_rate_front_list = [-0.2, -0.2, -0.2, -0.2]
    # args.expansion_rate_back_list = [0.1, 0.1, 0.1, 0.1]
    # args.audio_suffix = ".wav"
    args.segment_label_list = ["bwc1"]
    args.segment_sample_rate_list = [8000, 16000]
    args.segment_sample_rate_8k_name_list = []
    args.sample_rate = 16000
    args.expansion_rate_front_list = [-0.2, -0.2]
    args.expansion_rate_back_list = [0.1, 0.1]
    args.audio_suffix = ".wav"
    audio_lable_split(args)


if __name__ == "__main__":
    main()