import argparse
from auditok import split
import os
import pandas as pd

parser = argparse.ArgumentParser(description="Audio Split Using Auditok")
parser.add_argument('--audio_path', type=str,
                    default='E:\\project\\data\\speech\\kws\\XiaoYuNew\\11192020_0006.wav')
parser.add_argument('--output_dir', type=str,
                    default='E:\\project\\data\\speech\\kws\\XiaoYuNew')
# parser.add_argument('--output_format', type=str, default="RM_ROOM_Mandarin_S{:0>3d}M{}P{:0>5d}.wav")
# parser.add_argument('--output_format', type=str, default="RM_KWS_XIAORUI_{}_S{:0>3d}M{}P{}T{}.wav")
parser.add_argument('--output_format', type=str, default="RM_KWS_XIAOYU_{}_S{:0>3d}M{}P{}T{}.wav")
# parser.add_argument('--text', type=str, default="xiaorui,streamax,random")
parser.add_argument('--text', type=str, default="xiaoyu")
parser.add_argument('--speaker', type=int, default=7)
parser.add_argument('--sex', type=int, default=0, choices=[0, 1])
parser.add_argument('--language', type=int, default=0, choices=[0, 1])
parser.add_argument('--idx', type=int, default=1)
# parser.add_argument('--max_idx', type=int, default=10)
parser.add_argument('--max_idx', type=int, default=50)
args = parser.parse_args()

if __name__ == "__main__":
    audio_path = args.audio_path
    assert audio_path.endswith('.wav'), "[ERROR:] Only support wav data"

    output_path = os.path.join(
        args.output_dir, os.path.basename(audio_path).split('.')[0])
    assert not os.path.exists(
        output_path), "[ERROR:] Please remove directory: {}, firstly".format(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # init
    idx = args.idx
    text = args.text.split(',')
    text_idx = 0
    speaker = args.speaker
    audio_region_list = []

    # audio_regions = split(audio_path, 2, 10, 1.5, False, True)
    audio_regions = split(audio_path, 0.5, 10, 1.5, False, True)

    for region in audio_regions:
        audio_region_dict = {}
        output_name = args.output_format.format(
            text[text_idx], speaker, args.sex, args.language, idx)
        filename = region.save(os.path.join(output_path, output_name))
        audio_region_dict['audio_region'] = output_name.split('.')[0]
        audio_region_dict['state'] = 'N'
        audio_region_list.append(audio_region_dict)
        idx += 1
        print("Audio region saved as: {}".format(filename))

        if idx > args.max_idx:
            idx = args.idx
            text_idx += 1
        
        if text_idx == len(text):
            text_idx = 0
            speaker += 1


    audio_region_pd = pd.DataFrame(audio_region_list)
    audio_region_pd.to_csv(os.path.join(output_path, '{}.csv'.format(
        os.path.basename(audio_path).split('.')[0])), index=False)
