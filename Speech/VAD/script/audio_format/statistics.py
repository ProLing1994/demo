import argparse
import glob
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streamax Engine')
    # parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech/train-clean-360/")
    # parser.add_argument('--input_suffix', type=str, default=".flac")
    parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech_wav/train-clean-360/")
    parser.add_argument('--input_suffix', type=str, default=".wav")
    args = parser.parse_args()

    file_list = glob.glob(os.path.join(args.input_folder, '*/*/*' + args.input_suffix))
    file_list.sort()
    print(len(file_list))