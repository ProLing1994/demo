import argparse
import librosa
import os
import pandas as pd
import soundfile as sf

from tqdm import tqdm

def save_wav(wav, path, sampling_rate): 
    sf.write(path, wav, sampling_rate)


def get_sub_filepaths(folder):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
    return paths

def get_sub_filepaths_suffix(folder, suffix='.wav'):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            if not name.endswith(suffix):
                continue
            path = os.path.join(root, name)
            paths.append(path)
    return paths


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
    name_list = []
    id_name_list = []                               # [{'id':(), 'name':()}]
    try:
        id_name_pd = pd.read_csv(args.id_name_csv, encoding='utf_8_sig')

        name_list = id_name_pd["name"].to_list()
        name_list = [str(x) for x in name_list]

        for idx, row in id_name_pd.iterrows(): 
            id_name_list.append({'id': row['id'], 'name': row['name']}) 
    except:
        pass

    # id_name_test_list
    name_test_list = []
    try:
        id_name_test_pd = pd.read_csv(args.id_name_test_csv, encoding='utf_8_sig')
        name_test_list = id_name_test_pd["name"].to_list()
    except:
        pass

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
                segment_sample_rate = args.sample_rate
                expansion_fixed_samples = int(segment_sample_rate * args.expansion_fixed_length_s)

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
                        if args.bool_expansion_fixed_length:
                            data_length = int(line.split(':')[0].split('~')[1]) - int(line.split(':')[0].split('~')[0])
                            data_offset_front = (expansion_fixed_samples - data_length) // 2
                            data_offset_back = (expansion_fixed_samples - data_length) // 2 + (expansion_fixed_samples - data_length) % 2
                            audio_segments.append([max(0, int(line.split(':')[0].split('~')[0]) - data_offset_front), int(line.split(':')[0].split('~')[1]) + data_offset_back, time_id + line_idx])
                        else:
                            audio_segments.append([max(0, int(line.split(':')[0].split('~')[0]) + expansion_rate_front * segment_sample_rate), int(line.split(':')[0].split('~')[1]) + expansion_rate_back * segment_sample_rate, time_id + line_idx])
                f.close()

                # output audio_segment
                audio_data = librosa.core.load(audio_path, sr=args.sample_rate)[0]
                for segment_idx in range(len(audio_segments)):
                    audio_segment = audio_segments[segment_idx]
                    audio_segment_data = audio_data[int(audio_segment[0] * (args.sample_rate / segment_sample_rate)) : int(audio_segment[1] * (args.sample_rate / segment_sample_rate))]

                    if not len(audio_segment_data):
                        continue

                    # output 
                    output_dir = os.path.join(args.output_folder, "_".join(segment_label.split(' ')), equipment_name)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    output_path = os.path.join(output_dir, args.output_format.format(speaker_id, equipment_id, equipment_location, audio_segment[2]))
                    temp_path = os.path.join(args.output_folder, '{}{}'.format('temp', args.audio_suffix))
                    audio.save_wav(audio_segment_data.copy(), temp_path, args.sample_rate)
                    os.system('sox {} -b 16 -e signed-integer {}'.format(temp_path, output_path))

    id_name_pd = pd.DataFrame(id_name_list)
    id_name_pd.to_csv(args.id_name_csv, index=False, encoding="utf_8_sig")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    args = parser.parse_args()

    args.input_folder = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/collect/20230605/处理音频/"
    args.output_folder = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/collect/20230605/处理音频_out/"
    args.output_format = "RM_KWS_GORLIA_gorila_S{:0>3d}M0D{}{}T{:0>3d}.wav"
    args.id_name_csv = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/collect/20230605/唤醒词记录.csv"
    args.id_name_test_csv = "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/collect/20230605/唤醒词记录_测试人员.csv"

    # 寻找相同说话人音频，记录末尾编号，新增数据向后延续
    args.find_folder_list = [
                                "/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/collect/20230605/处理音频_out/",
                            ]

    # params
    args.equipment_name_list = ['mic_130cm', 'phone', 'adplus1_0_normal', 'adplus1_0_70cm', 'adplus1_0_100cm', 'adplus2_0_normal', 'adplus2_0_70cm', 'adplus2_0_100cm']
    args.equipment_id_list = [1, 5, 4, 4, 4, 8, 8, 8]
    args.equipment_location_list = [4, 0, 5, 3, 4, 5, 3 ,4]

    args.segment_label_list = ["Start Recording"]
    args.sample_rate = 16000
    
    # 是否将音频扩展为固定的长度
    # args.bool_expansion_fixed_length = True
    args.bool_expansion_fixed_length = False
    args.expansion_fixed_length_s = 3.0

    # 若 args.expansion_fixed_length = True，下述无效
    args.expansion_rate_front_list = [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2]
    args.expansion_rate_back_list = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ]
    args.audio_suffix = ".wav"

    audio_lable_split(args)