# 注：请执行操作顺序：
# 1. 执行本文件，根据 txt 文件和 wav 文件自动截取音频，保存截取后的音频（该 txt 文件来自音频裁剪工具）

import argparse
import glob
import librosa
import os 
import pandas as pd
import wave
import yaml

from tqdm import tqdm

def main():
    # mkdir 
    if not os.path.exists(os.path.join(args.output_folder, args.label)):
        os.makedirs(os.path.join(args.output_folder, args.label))

    # load audio_list 
    file_list = []
    audio_list = []
    audio_list = glob.glob(os.path.join(args.input_folder, '*' + args.audio_suffix))
    audio_list.sort()

    for idx in tqdm(range(len(audio_list))):
        wave_path = audio_list[idx]

        label_path = wave_path.replace(args.audio_suffix, args.label_suffix)
        if not os.path.exists(label_path):
            continue

        # init 
        single_segments = []
        continuous_segments = []
        wave_params = wave.open(wave_path)
        sample_rate = wave_params.getframerate()   # 获得采样率

        f = open(label_path, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            if 'single' in line:
                single_segments.append([int(line.split(':')[0].split('~')[0]), int(line.split(':')[0].split('~')[1])])
            if 'continue' in line:
                continuous_segments.append([int(line.split(':')[0].split('~')[0]), int(line.split(':')[0].split('~')[1])])
        f.close()

        audio_data = librosa.core.load(wave_path, sr=sample_rate)[0]

        for segment_idx in range(len(single_segments)):
            single_segment = single_segments[segment_idx]
            single_data = audio_data[int(single_segment[0]) : int(single_segment[1])]
        
            # output 
            output_path = os.path.join(args.output_folder, args.label, args.dataset_name + args.label + "_{}_{}_{:0>3d}{}".format(os.path.basename(wave_path).split('.')[0], 'Single', segment_idx, args.audio_suffix))
            librosa.output.write_wav(output_path, single_data, sr=sample_rate) 
            file_list.append({"input path": wave_path, "output": output_path})
        
        for segment_idx in range(len(continuous_segments)):
            continuous_segment = continuous_segments[segment_idx]
            continuous_data = audio_data[int(continuous_segment[0]) : int(continuous_segment[1])]
 
            # output
            output_path = os.path.join(args.output_folder, args.label, args.dataset_name + args.label + "_{}_{}_{:0>3d}{}".format(os.path.basename(wave_path).split('.')[0], 'Auto', segment_idx, args.audio_suffix))
            librosa.output.write_wav(output_path, continuous_data, sr=sample_rate) 
            file_list.append({"input path": wave_path, "output": output_path})

    file_pd = pd.DataFrame(file_list)
    file_pd.to_csv(os.path.join(args.output_folder, args.label, args.label + '.csv'), index=False, encoding="utf_8_sig")

if __name__ == "__main__":
    # default_input_folder = "/mnt/huanyuan/data/speech/sed/TUT-Rare-Sound-Events-2017/original_dataset/TUT-rare-sound-events-2017-evaluation/data/source_data/events/gunshot_rm_labeled/"
    # default_output_folder = "/mnt/huanyuan/data/speech/sed/TUT-Rare-Sound-Events-2017/processed_dataset/"
    # default_dataset_name = "TUT2017_SED_"
    # default_label = "gunshot_rm_labeled"

    default_input_folder = "/mnt/huanyuan/data/speech/sed/FSD50K/processed_dataset/Total_Gunshot_and_gunfire_rm_labeled/"
    default_output_folder = "/mnt/huanyuan/data/speech/sed/FSD50K/processed_dataset/"
    default_dataset_name = "FSD50K_SED_"
    default_label = "gunshot_rm_labeled"

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_folder', type=str, default=default_input_folder)
    parser.add_argument('--output_folder', type=str, default=default_output_folder)
    parser.add_argument('--dataset_name', type=str, default=default_dataset_name)
    parser.add_argument('--label', type=str, default=default_label)
    parser.add_argument('--audio_suffix', type=str, default=".wav")
    parser.add_argument('--label_suffix', type=str, default=".txt")
    args = parser.parse_args()

    main()