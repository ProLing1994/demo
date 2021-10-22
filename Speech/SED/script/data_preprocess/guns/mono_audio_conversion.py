import argparse
import glob
import librosa
import os 
import sys

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio


def main():
    # init
    sample_rate = 44100

    # mkdir 
    if not os.path.exists(os.path.join(args.output_folder)):
        os.makedirs(os.path.join(args.output_folder))

    # load audio_list 
    audio_list = []
    audio_list = glob.glob(os.path.join(args.input_folder, '*' + args.audio_suffix))
    audio_list.sort()

    for idx in tqdm(range(len(audio_list))):
        wave_path = audio_list[idx]
        audio_data = librosa.core.load(wave_path, sr=sample_rate)[0]
        
        # output
        temp_path = os.path.join(args.output_folder, "temp.wav")
        output_path = wave_path.replace(args.input_folder, args.output_folder)
        audio.save_wav(audio_data.copy(), temp_path, sample_rate)
        script = " ".join(['sox', temp_path, '-c 1', '-b 16', '-r', str(sample_rate), '-e signed-integer', output_path])
        os.system(script)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TUT-Rare-Sound-Events-2017 Engine')
    # parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/sed/TUT-Rare-Sound-Events-2017/original_dataset/TUT-rare-sound-events-2017-evaluation/data/source_data/events/gunshot/")
    # parser.add_argument('--output_folder', type=str, default="/mnt/huanyuan/data/speech/sed/TUT-Rare-Sound-Events-2017/original_dataset/TUT-rare-sound-events-2017-evaluation/data/source_data/events/gunshot_rm/")
    parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/sed/FSD50K/processed_dataset/Total_Gunshot_and_gunfire/")
    parser.add_argument('--output_folder', type=str, default="/mnt/huanyuan/data/speech/sed/FSD50K/processed_dataset/Total_Gunshot_and_gunfire_rm/")
    parser.add_argument('--audio_suffix', type=str, default=".wav")
    args = parser.parse_args()

    main()