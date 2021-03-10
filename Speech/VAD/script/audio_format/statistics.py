import argparse
import glob
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streamax Engine')
    # parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech/train-other-500/")
    # parser.add_argument('--input_suffix', type=str, default=".flac")
    # parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech_wav/train-clean-360/")
    parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech_wav_record/train-clean-360/")
    parser.add_argument('--input_suffix', type=str, default=".wav")
    # parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/sed/FSD50K/processed_dataset/leaf/")
    # parser.add_argument('--input_suffix', type=str, default=".wav")
    args = parser.parse_args()

    file_list = glob.glob(os.path.join(args.input_folder, '*' + args.input_suffix))
    file_list += glob.glob(os.path.join(args.input_folder, '*/*' + args.input_suffix))
    file_list += glob.glob(os.path.join(args.input_folder, '*/*/*' + args.input_suffix))
    file_list.sort()
    print(len(file_list))