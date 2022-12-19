import argparse
import librosa
import os
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio


def audio_lable_split(args):
    # file list 
    file_list = os.listdir(args.input_folder)
    file_list.sort()

    # output_folder
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    # id_name
    speaker_id = args.start_speaker_id

    for idx in tqdm(range(len(file_list))):
        if not file_list[idx].endswith('.txt'):
            continue
        
        # init 
        label_path = os.path.join(args.input_folder, file_list[idx])
        audio_path = label_path.split('.')[0] + args.audio_suffix

        audio_segments = []
        f = open(label_path, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            wav_id = int(line.split(':')[-1]) 
            audio_segments.append([int(line.split(':')[0].split('~')[0]) + args.expansion_rate_front * args.sample_rate, \
                                     int(line.split(':')[0].split('~')[1]) + args.expansion_rate_back * args.sample_rate, \
                                     wav_id])
        f.close()

        # output audio_segment
        # audio_data, _ = sf.read(audio_path)
        audio_data = librosa.core.load(audio_path, sr=args.sample_rate)[0]
        for segment_idx in range(len(audio_segments)):
            audio_segment = audio_segments[segment_idx]
            audio_segment_data = audio_data[int(audio_segment[0]) : int(audio_segment[1])]

            # output 
            output_path = os.path.join(args.output_folder, args.output_format.format(speaker_id, audio_segment[2]))
            temp_path = os.path.join(args.output_folder, '{}{}'.format('temp', args.audio_suffix))
            audio.save_wav(temp_path, audio_segment_data, args.sample_rate)
            os.system('sox {} -b 16 -e signed-integer {}'.format(temp_path, output_path))

        speaker_id += 1
        
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/Recording/temp/0831") 
    parser.add_argument('--output_folder', type=str, default="/mnt/huanyuan/data/speech/Recording/temp/0831_out") 
    parser.add_argument('--output_format', type=str, default="RM_Foreigner_English_BWC_S{:0>3d}P{:0>3d}.wav")
    parser.add_argument('--start_speaker_id', type=int, default=10)
    args = parser.parse_args()

    # params
    args.sample_rate = 16000
    args.audio_suffix = ".wav"
    args.expansion_rate_front = -0.1
    args.expansion_rate_back = 0.0
    audio_lable_split(args)


if __name__ == "__main__":
    main()