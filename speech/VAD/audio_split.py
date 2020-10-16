import argparse
from auditok import split
import os
import pandas as pd 

parser = argparse.ArgumentParser(description="Audio Split Using Auditok")
parser.add_argument('--audio_path', type=str, default='E:\\project\\data\\weiboyulu\\1012\\0000000000000000-201012-103547-114012-000001089960.wav')
parser.add_argument('--output_dir', type=str, default='E:\\project\\data\\weiboyulu\\1012')
parser.add_argument('--speaker', type=int, default=1)
parser.add_argument('--sex', type=int, default=0, choices=[0, 1])
parser.add_argument('--idx', type=int, default=1)
args = parser.parse_args()

if __name__ == "__main__":
  audio_path = args.audio_path
  assert audio_path.endswith('.wav'), "[ERROR:] Only support wav data"

  output_path = os.path.join(args.output_dir, os.path.basename(audio_path).split('.')[0])
  assert not os.path.exists(output_path), "[ERROR:] Please remove directory: {}, firstly".format(output_path)

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  idx = args.idx
  output_file = "RM_ROOM_Mandarin_S{:0>3d}M{}".format(args.speaker, args.sex)
  audio_regions = split(audio_path, 2, 10, 1.5, False, True)
  audio_region_list = []
  for region in audio_regions:
    audio_region_dict = {}
    filename = region.save(os.path.join(output_path, output_file + "P{:0>5d}.wav".format(idx)))
    audio_region_dict['audio_region'] = output_file + "P{:0>5d}".format(idx)
    audio_region_dict['state'] = 'N'
    audio_region_list.append(audio_region_dict)
    idx += 1
    print("Audio region saved as: {}".format(filename))
  audio_region_pd = pd.DataFrame(audio_region_list)
  audio_region_pd.to_csv(os.path.join(output_path, '{}.csv'.format(os.path.basename(audio_path).split('.')[0])), index=False)
