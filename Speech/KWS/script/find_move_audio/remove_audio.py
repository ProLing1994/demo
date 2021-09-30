import argparse
import os
import sys

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/common/common')
from utils.python.folder_tools import *

def remove_audio_samename():
    wave_list = os.listdir(args.input_dir)

    for idx in tqdm(range(len(wave_list))):
        wave_path = os.path.join(args.input_dir, wave_list[idx])
        output_path = os.path.join(args.output_dir, os.path.basename(wave_path).split('.')[0] + '.wav')

        if os.path.exists(output_path):
            print(output_path)
            # os.remove(output_path)


def remove_audio_suffix():
    path_list = get_sub_filepaths_suffix(args.input_dir, args.suffix)

    for idx in tqdm(range(len(path_list))):
        if os.path.exists(path_list[idx]):
            print(path_list[idx])
            # os.remove(path_list[idx])


if __name__ == "__main__":

    # 方法：删除同名文件
    parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    args = parser.parse_args()
    args.input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/KwsEnglishDataset/tts/sv2tts/LibriSpeech/over_short/"
    args.output_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/KwsEnglishDataset/tts/sv2tts/LibriSpeech/train-other-500_check_vad/activatebwc/"

    remove_audio_samename()

    # # 方法：删除制定后缀
    # parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    # args = parser.parse_args()
    # args.input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/original_dataset/ActivateBWC_07162021/activatebwc/原始音频/"
    # args.suffix = '_temp.wav'

    # remove_audio_suffix()