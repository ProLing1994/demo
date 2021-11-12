import argparse
import librosa
import os

from tqdm import tqdm

def get_sub_filepaths_suffix(folder, suffix='.wav'):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            if not name.endswith(suffix):
                continue
            path = os.path.join(root, name)
            paths.append(path)
    return paths


def static_audio_number(args):
    file_list = get_sub_filepaths_suffix(args.input_folder, args.input_suffix)
    file_list.sort()
    print("Audio Number: ", len(file_list))


def static_audio_length(args):
    file_list = get_sub_filepaths_suffix(args.input_folder, args.input_suffix)
    file_list.sort()

    # init 
    audio_length_num = 0

    for idx in tqdm(range(len(file_list))):
        file_name = file_list[idx]

        if not file_name.endswith(args.input_suffix):
            continue

        audio_path = os.path.join(args.input_folder, file_name)
        audio_data = librosa.core.load(audio_path, sr=args.sample_rate)[0]
        audio_length = int(len(audio_data) * 1000 / args.sample_rate)
        audio_length_num += audio_length
    
    print("Audio Length: {}ms({:.3f}h)".format(audio_length_num, audio_length_num/1000/3600))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streamax Engine')
    args = parser.parse_args()
    args.input_folder = "/mnt/huanyuan/data/speech/Recording/MTA_Truck_Platformalarm/adpro_8k/1108_1110/原始音频/"
    args.input_suffix = ".wav"
    args.sample_rate = 8000

    static_audio_number(args)

    print("[Begin] Static Audio Length")
    static_audio_length(args)
    print("[Done] Static Audio Length")