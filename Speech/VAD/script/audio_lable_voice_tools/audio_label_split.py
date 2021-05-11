import argparse
import librosa
import os
import pandas as pd
import wave 

from tqdm import tqdm


def audio_lable_csv(args):
    # id_name_list
    id_name_pd = pd.read_csv(args.id_name_csv, encoding='utf_8_sig')
    name_list = id_name_pd["name"].to_list()
    id_name_list = []                               # [{'id':(), 'name':()}]
    for idx, row in id_name_pd.iterrows(): 
        id_name_list.append({'id': row['id'], 'name': row['name']}) 

    # file list 
    file_list = os.listdir(args.input_folder)
    file_list.sort()

    for idx in tqdm(range(len(file_list))):
        if not file_list[idx].endswith('.txt'):
            continue
        
        # init 
        label_path = os.path.join(args.input_folder, file_list[idx])
        audio_path = label_path.split('.')[0] + args.audio_suffix

        # temp_path = os.path.join(args.input_folder, "temp.wav")
        # os.system('sox {} -b 16 -e signed-integer {}'.format(audio_path, temp_path))
        # wave_params = wave.open(temp_path)
        # sample_rate = wave_params.getframerate()   # 获得采样率

        # id_name
        audio_name = file_list[idx].split('.')[0]
        if audio_name in name_list:
            speaker_id = id_name_pd[id_name_pd["name"] == audio_name]["id"].to_list()[0]
        else:
            speaker_id = len(id_name_list) + 1
            id_name_list.append({'id': speaker_id, 'name': audio_name}) 

        audio_segments = []
        f = open(label_path, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            audio_segments.append([int(line.split(':')[0].split('~')[0]), int(line.split(':')[0].split('~')[1])])
        f.close()

        # output audio_segment
        audio_data = librosa.core.load(audio_path, sr=args.sample_rate)[0]
        for segment_idx in range(len(audio_segments)):
            audio_segment = audio_segments[segment_idx]
            audio_segment_data = audio_data[int(audio_segment[0]) : int(audio_segment[1])]

            # output 
            output_path = os.path.join(args.output_folder, args.output_format.format(speaker_id, segment_idx + 1))
            temp_path = os.path.join(args.output_folder, '{}{}'.format('temp', args.audio_suffix))
            librosa.output.write_wav(temp_path, audio_segment_data, sr=args.sample_rate) 
            os.system('sox {} -b 16 -e signed-integer {}'.format(temp_path, output_path))

    id_name_pd = pd.DataFrame(id_name_list)
    id_name_pd.to_csv(args.id_name_csv, index=False, encoding="utf_8_sig")

        
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/kws/xiaorui_dataset/original_dataset/XiaoRuiDataset_05102021/已标注/") 
    parser.add_argument('--output_folder', type=str, default="/mnt/huanyuan/data/speech/kws/xiaorui_dataset/original_dataset/XiaoRuiDataset_05102021/xiaorui/") 
    parser.add_argument('--output_format', type=str, default="RM_KWS_XIAORUI_xiaorui_S{:0>3d}M0D51T{:0>3d}.wav")
    parser.add_argument('--id_name_csv', type=str, default="/mnt/huanyuan/data/speech/kws/xiaorui_dataset/original_dataset/唤醒词记录.csv") 
    args = parser.parse_args()

    # params
    args.sample_rate = 16000
    args.audio_suffix = ".wav"
    audio_lable_csv(args)


if __name__ == "__main__":
    main()