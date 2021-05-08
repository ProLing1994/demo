import argparse
import librosa
import os 
import pandas as pd
import sys

from tqdm import tqdm

def static_audio_length(args):
    file_list = os.listdir(args.input_folder)
    file_list.sort()

    # init 
    audio_length_num = 0

    for idx in tqdm(range(len(file_list))):
        file_name = file_list[idx]

        if not file_name.endswith('.wav'):
            continue
            
        # satisfy file_format
        bool_satisfy_file_format = True
        for file_format in args.file_format_list:
            if file_format not in file_name:
                bool_satisfy_file_format = False
        if not bool_satisfy_file_format:
            continue

        audio_path = os.path.join(args.input_folder, file_name)
        audio_data = librosa.core.load(audio_path, sr=args.sample_rate)[0]
        audio_length = int(len(audio_data) * 1000 / args.sample_rate)
        audio_length_num += audio_length
    
    print("Audio Length: {}ms({:.3f}h)".format(audio_length_num, audio_length_num/1000/3600))

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # 16k
    # args.input_folder = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/test_dataset/海外同事录制_0425/办公室场景/场景二/"
    # args.file_format_list = ["RM_KWS_ACTIVATEBWC_ovweseas_asr_"]
    # args.sample_rate = 16000

    # 8k
    args.input_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/其他录音/adpro-2"
    args.file_format_list = [".wav"]
    # args.input_folder = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/实车场景/处理音频/"
    # args.file_format_list = ["danbin_asr"]
    args.sample_rate = 8000
    
    print("[Begin] Static Audio Length")
    static_audio_length(args)
    print("[Done] Static Audio Length")


if __name__ == "__main__":
    main() 