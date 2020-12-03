import argparse
from auditok import split
import os
import pandas as pd

parser = argparse.ArgumentParser(description="Audio Split Using Auditok")
parser.add_argument('--audio_path', type=str,
                    default='D:\\data\\test\\5.wav')
parser.add_argument('--output_dir', type=str,
                    default='D:\\data\\test')
parser.add_argument('--output_format', type=str, default="RM_ROOM_Mandarin_S{:0>3d}M{}P{:0>5d}.wav")
# parser.add_argument('--output_format', type=str, default="RM_KWS_XIAORUI_{}_S{:0>3d}M{}D{}T{}.wav")
# parser.add_argument('--text', type=str, default="xiaorui")
parser.add_argument('--speaker', type=int, default=6)
parser.add_argument('--sex', type=int, default=0, choices=[0, 1])
# parser.add_argument('--distance', type=int, default=0, choices=[0, 1, 2])
parser.add_argument('--idx', type=int, default=1)
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
    audio_region_list = []

    # audio_regions = split(audio_path, 2, 10, 1.5, False, True)
    audio_regions = split(audio_path, 2, 10, 1.0, False, True, energy_threshold=55)
    # audio_regions = split(audio_path, 1, 5, 0.05, False, True)

    for region in audio_regions:
        audio_region_dict = {}
        output_name = args.output_format.format(
            args.speaker, args.sex, idx)
        # output_name = args.output_format.format(
        #     args.text, args.speaker, args.sex, args.distance, idx)
        filename = region.save(os.path.join(output_path, output_name))
        audio_region_dict['audio_region'] = output_name.split('.')[0]
        audio_region_dict['state'] = 'N'
        audio_region_list.append(audio_region_dict)
        idx += 1
        print("Audio region saved as: {}".format(filename))


    audio_region_pd = pd.DataFrame(audio_region_list)
    audio_region_pd.to_csv(os.path.join(output_path, '{}.csv'.format(
        os.path.basename(audio_path).split('.')[0])), index=False)
