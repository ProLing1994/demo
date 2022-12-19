import argparse
import librosa
import os 
import pandas as pd
import sys

from tqdm import tqdm

def static_audio_length(args):
    # load csv
    dataset_pd = pd.read_csv(args.csv_path)
    dataset_pd = dataset_pd[dataset_pd["type"] == args.type]
    dataset_pd = dataset_pd[dataset_pd["bool_noise_reduction"] == args.bool_noise_reduction]

    # init 
    sample_rate = 16000
    audio_length_list = []
    audio_length_num = 0
    for _, row in tqdm(dataset_pd.iterrows()):
        audio_path = row['path']
        audio_data = librosa.core.load(audio_path, sr=sample_rate)[0]
        audio_length = int(len(audio_data) * 1000 / sample_rate)
        audio_length_list.append(audio_length)
        audio_length_num += audio_length
    
    print("Audio Length: {}ms({:.3f}h)".format(audio_length_num, audio_length_num/1000/3600))

def main():
    # config file
    default_csv_path = "/mnt/huanyuan/data/speech/Real_vehicle_sample/20201218/Real_vehicle_sample_20201218.csv"
    default_type = 'normal_driving'                 # ['normal_driving', 'idling_driving']
    default_bool_noise_reduction = False            # [False, True]

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default=default_csv_path)
    parser.add_argument('--type', type=str, default=default_type)
    parser.add_argument('--bool_noise_reduction', type=bool, default=default_bool_noise_reduction)
    args = parser.parse_args()

    print("[Begin] Static Audio Length")
    static_audio_length(args)
    print("[Done] Static Audio Length")


if __name__ == "__main__":
    main()