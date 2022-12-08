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
    sample_rate = 16000

    # mkdir 
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # load audio_list
    audio_list = []
    print("[Init:] Load audio list: ")
    wave_list = glob.glob(os.path.join(args.record_folder, '*' + args.suffix))
    wave_list += glob.glob(os.path.join(args.record_folder, '*/*' + args.suffix))
    wave_list += glob.glob(os.path.join(args.record_folder, '*/*/*' + args.suffix))
    wave_list += glob.glob(os.path.join(args.record_folder, '*/*/*/*' + args.suffix))
    for idx in tqdm(range(len(wave_list))):
        wave_path = wave_list[idx]

        if not wave_path.endswith(args.suffix):
            continue

        # output_path = wave_path.replace(args.record_folder, args.output_folder)
        # if os.path.exists(output_path):
        #     continue

        audio_list.append(wave_path)

    audio_list.sort()
    # Find error audio data
    print("[Init:] Find error audio data: ")
    for idx in tqdm(range(len(audio_list))):
        wave_path = audio_list[idx]
        data = librosa.core.load(wave_path, sr=sample_rate)[0]

        if  len(data) == 0 or (data.max() * (pow(2,15)) == 16349 and data.min() * (pow(2,15)) == -16350.0) :
            output_path = wave_path.replace(args.record_folder, args.output_folder)
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if mode == 0:
                input_path = wave_path.replace(args.record_folder, args.input_folder)
                os.system("cp {} {}".format(input_path, output_path))
            elif mode == 1:
                audio.save_wav(data, output_path, sample_rate)
            elif mode == 2:
                print(wave_path)
                os.remove(wave_path)
            else:
                raise Exception("[ERROR: ] Unknow mode")

if __name__ == "__main__":
    # mode: [0, 1]
    # 0: 拷贝原始音频
    # 1：拷贝错误音频
    # 2: 删除错误音频
    mode = 1

    parser = argparse.ArgumentParser(description='Streamax Engine')
    parser.add_argument('--record_folder', type=str, default="/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech_wav_record/")
    parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech_wav/")
    parser.add_argument('--output_folder', type=str, default="/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech_wav_record_error_found/")
    parser.add_argument('--suffix', type=str, default=".wav")
    args = parser.parse_args()

    main()