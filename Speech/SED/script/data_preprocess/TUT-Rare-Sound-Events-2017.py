# 注：请执行操作顺序：
# 1. 执行本文件，根据 yaml 文件和 wav 文件自动截取音频，保存截取后的音频

import argparse
import glob
import librosa
import os 
import pandas as pd
import yaml

from tqdm import tqdm


def main():
    # mkdir 
    if not os.path.exists(os.path.join(args.output_folder, args.label)):
        os.makedirs(os.path.join(args.output_folder, args.label))

    # load audio_list 
    file_list = []
    audio_list = []
    audio_list = glob.glob(os.path.join(args.input_folder, args.label, '*' + args.audio_suffix))
    audio_list.sort()

    for idx in tqdm(range(len(audio_list))):
        wave_path = audio_list[idx]

        label_path = wave_path.replace(args.audio_suffix, args.label_suffix)
        if not os.path.exists(label_path):
            continue

        f = open(label_path, 'r', encoding='utf-8')
        info_yaml = yaml.load(f.read(), Loader=yaml.FullLoader)

        sample_rate = info_yaml['samplerate']
        single_segments = []
        continuous_segments = []
        if 'valid_segments' in info_yaml:
            single_segments += info_yaml['valid_segments']
        if 'single_segments' in info_yaml and info_yaml['single_segments']:
            single_segments += info_yaml['single_segments']
        if 'continuous_segments' in info_yaml and info_yaml['continuous_segments']:
            continuous_segments += info_yaml['continuous_segments']

        audio_data = librosa.core.load(wave_path, sr=sample_rate)[0]

        for segment_idx in range(len(single_segments)):
            single_segment = single_segments[segment_idx]
            single_data = audio_data[int((single_segment[0] * sample_rate)) : int((single_segment[1]) * sample_rate)]
        
            # output 
            output_path = os.path.join(args.output_folder, args.label, "TUT2017_SED_" + args.label + "_{}_{}_{:0>3d}{}".format(os.path.basename(wave_path).split('.')[0], 'Single', segment_idx, args.audio_suffix))
            librosa.output.write_wav(output_path, single_data, sr=sample_rate) 
            file_list.append({"input path": wave_path, "output": output_path})
        
        for segment_idx in range(len(continuous_segments)):
            continuous_segment = continuous_segments[segment_idx]
            continuous_data = audio_data[int((continuous_segment[0] * sample_rate)) : int((continuous_segment[1]) * sample_rate)]

            # output
            output_path = os.path.join(args.output_folder, args.label, "TUT2017_SED_" + args.label + "_{}_{}_{:0>3d}{}".format(os.path.basename(wave_path).split('.')[0], 'Auto', segment_idx, args.audio_suffix))
            librosa.output.write_wav(output_path, continuous_data, sr=sample_rate) 
            file_list.append({"input path": wave_path, "output": output_path})

    file_pd = pd.DataFrame(file_list)
    file_pd.to_csv(os.path.join(args.output_folder, args.label, args.label + '_1.csv'), index=False, encoding="utf_8_sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TUT-Rare-Sound-Events-2017 Engine')
    parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/sed/TUT-Rare-Sound-Events-2017/original_dataset/TUT-rare-sound-events-2017-development/data/source_data/events/")
    # parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/sed/TUT-Rare-Sound-Events-2017/original_dataset/TUT-rare-sound-events-2017-evaluation/data/source_data/events/")
    parser.add_argument('--output_folder', type=str, default="/mnt/huanyuan/data/speech/sed/TUT-Rare-Sound-Events-2017/processed_dataset/")
    # parser.add_argument('--label', type=str, default="gunshot")
    parser.add_argument('--label', type=str, default="gunshot_rm_labeled")
    parser.add_argument('--audio_suffix', type=str, default=".wav")
    parser.add_argument('--label_suffix', type=str, default=".yaml")
    args = parser.parse_args()

    main()