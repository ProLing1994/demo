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


def audio_lable_split(args):
    # id_name_list
    id_name_list = []                               # [{'id':(), 'name':()}]
    try:
        id_name_pd = pd.read_csv(args.id_name_csv, encoding='utf_8_sig')

        for idx, row in id_name_pd.iterrows(): 
            id_name_list.append({'id': row['id'], 'name': row['name']}) 
    except:
        pass

    # file list 
    file_list = get_sub_filepaths(args.input_folder)
    file_list.sort()

    for idx in tqdm(range(len(file_list))):

        # txt 文件
        if file_list[idx].endswith('.txt'):

            # label path
            label_path = file_list[idx]

            # speaker_name
            speaker_name = os.path.basename(label_path).split('.')[0]
            
            # speaker_id
            speaker_id = len(id_name_list) + 1
            id_name_list.append({'id': speaker_id, 'name': speaker_name}) 

            # init
            time_id = 0
            
            # 遍历标签名字
            for segment_label_idx in range(len(args.segment_label_list)):

                # 标签名字
                segment_label = args.segment_label_list[segment_label_idx]

                for equipment_idx in range(len(args.equipment_name_list)):

                    segment_sample_rate = args.sample_rate
                    expansion_rate_front = args.expansion_rate_front_list[equipment_idx]
                    expansion_rate_back = args.expansion_rate_back_list[equipment_idx]

                    # equipment info
                    equipment_name = args.equipment_name_list[equipment_idx]
                    equipment_id = args.equipment_id_list[equipment_idx]
                    equipment_location = args.equipment_location_list[equipment_idx]

                    # audio path
                    audio_path = label_path.split('.')[0] + '_' + equipment_name + args.audio_suffix

                    # 音频不存在，则不进行截取
                    if not os.path.exists(audio_path):
                        continue
                    
                    # audio segments
                    # 加载标签文件，添加音频分割列表
                    audio_segments = []         # [起点， 终点， 序号]
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

                        # 音频数据
                        audio_segment_data = audio_data[int(audio_segment[0] * (args.sample_rate / segment_sample_rate)) : int(audio_segment[1] * (args.sample_rate / segment_sample_rate))]

                        if not len(audio_segment_data):
                            continue

                        # output 
                        output_dir = os.path.join(args.output_folder, "_".join(segment_label.split(' ')), equipment_name)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        # 保存音频
                        output_path = os.path.join(output_dir, args.output_format.format("_".join(segment_label.split(' ')), speaker_id, args.male, equipment_id, equipment_location, audio_segment[2]))
                        temp_path = os.path.join(args.output_folder, '{}{}'.format('temp', args.audio_suffix))
                        save_wav(audio_segment_data.copy(), temp_path, args.sample_rate)
                        os.system('sox {} -b 16 -e signed-integer {}'.format(temp_path, output_path))

    # id_name_list
    id_name_pd = pd.DataFrame(id_name_list)
    id_name_pd.to_csv(args.id_name_csv, index=False, encoding="utf_8_sig")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/collect/20230605/test_out/")
    parser.add_argument('--output_folder', type=str, default="/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/collect/20230605/test_res/")
    parser.add_argument('--id_name_csv', type=str, default="/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/collect/20230605/唤醒词记录.csv")
    parser.add_argument('--output_format', type=str, default="RM_KWS_GORLIA_{}_S{:0>3d}M{}D{}{}T{:0>3d}.wav")
    parser.add_argument('--male', type=int, default=0)  # 女 0   男 1
    parser.add_argument('--equipment_name_list', type=str, default="mic_130cm,phone,adplus1_0_normal,adplus1_0_70cm,adplus1_0_100cm,adplus2_0_normal,adplus2_0_70cm,adplus2_0_100cm")
    parser.add_argument('--equipment_id_list', type=str, default="1,5,4,4,4,8,8,8")
    parser.add_argument('--equipment_location_list', type=str, default="4,0,5,3,4,5,3,4")
    parser.add_argument('--segment_label_list', type=str, default="gorila_gorila")
    parser.add_argument('--sample_rate', type=int, default=16000)
    args = parser.parse_args()

    # params
    args.equipment_name_list = args.equipment_name_list.split(',')
    args.equipment_id_list = args.equipment_id_list.split(',')
    args.equipment_location_list = args.equipment_location_list.split(',')

    # 标签名字
    args.segment_label_list = args.segment_label_list.split(',')
    
    args.expansion_rate_front_list = [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2]
    args.expansion_rate_back_list = [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ]
    args.audio_suffix = ".wav"

    audio_lable_split(args)