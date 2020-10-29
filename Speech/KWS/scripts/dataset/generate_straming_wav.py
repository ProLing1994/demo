import argparse
import librosa
import numpy as np
import os 
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import load_cfg_file
from dataset.kws.dataset_helper import *


def mix_in_audio_sample(track_data, track_offset, sample_data, sample_offset,
                        clip_duration, sample_volume, ramp_in, ramp_out):
    """Mixes the sample data into the main track at the specified offset.

    Args:
        track_data: Numpy array holding main audio data. Modified in-place.
        track_offset: Where to mix the sample into the main track.
        sample_data: Numpy array of audio data to mix into the main track.
        sample_offset: Where to start in the audio sample.
        clip_duration: How long the sample segment is.
        sample_volume: Loudness to mix the sample in at.
        ramp_in: Length in samples of volume increase stage.
        ramp_out: Length in samples of volume decrease stage.
    """
    ramp_out_index = clip_duration - ramp_out
    track_end = min(track_offset + clip_duration, track_data.shape[0])
    track_end = min(track_end, track_offset + (sample_data.shape[0] - sample_offset))
    sample_range = track_end - track_offset
    for i in range(sample_range):
        if i < ramp_in:
            envelope_scale = i / ramp_in
        elif i > ramp_out_index:
            envelope_scale = (clip_duration - i) / ramp_out
        else:
            envelope_scale = 1
        sample_input = sample_data[sample_offset + i]
        track_data[track_offset + i] += sample_input * envelope_scale * sample_volume


def straming_dataset_generator(input_dir, output_path, config_file, add_noise_on, mode, test_duration_seconds, word_gap_ms):
    # load configuration file
    cfg = load_cfg_file(config_file)
    # init
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    output_audio_sample_count = sample_rate * test_duration_seconds
    output_audio = np.zeros((output_audio_sample_count,), dtype=np.float32)
    word_gap_samples = int((word_gap_ms * sample_rate) / 1000)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # mode
    audio_list = [] # {'file': ..., 'lable': ...}
    if mode == 0:
        print("Generator Straming Dataset From Config File")
        data_pd = pd.read_csv(cfg.general.data_csv_path)
        data_pd = data_pd[data_pd['mode'] == 'testing']
        # data_pd = data_pd[data_pd['mode'] == 'training']
        for _, row in data_pd.iterrows():
            audio_dict = {}
            audio_dict['file'] = row['file']
            audio_dict['lable'] = row['label']
            audio_list.append(audio_dict)
    elif mode == 1:
        print("Generator Straming Dataset From Folder")
        file_list = os.listdir(input_dir)
        for file in file_list:
            audio_dict = {}
            audio_dict['file'] = os.path.join(input_dir, file)
            audio_dict['lable'] = UNKNOWN_WORD_LABEL
            audio_list.append(audio_dict)
    else:
        pass

    # Mix the background audio into the main track.
    if add_noise_on:
        pass

    # Mix the audio into the main track, noting their labels and positions.
    output_file_list = [] # {'file': ..., 'lable': ..., 'start_time': ..., 'end_time': ...}
    output_offset = 0
    while(output_offset < output_audio_sample_count):
        output_offset += word_gap_samples + np.random.randint(word_gap_samples)
        output_offset_ms = (output_offset * 1000) / sample_rate

        if mode == 0:
            data_index = np.random.randint(len(audio_list))
            found_data = audio_list[data_index]['file']
            found_label = audio_list[data_index]['lable']
        elif mode == 1:
            data_index = np.random.randint(len(audio_list))
            found_data = audio_list[data_index]['file']
            found_label = audio_list[data_index]['lable']
        else:
            pass 
        
        # load data
        if found_label == SILENCE_LABEL:
            found_audio = np.zeros(desired_samples, dtype=np.float32)
        else:
            found_audio = librosa.core.load(found_data, sr=sample_rate)[0]

        found_audio_length = len(found_audio)
        mix_in_audio_sample(output_audio, output_offset, found_audio, 0, found_audio_length, 1.0, 500, 500)
        output_offset += found_audio_length

        output_file_dict = {}
        output_file_dict['file'] = found_data
        output_file_dict['lable'] = found_label
        output_file_dict['start_time'] = output_offset_ms
        output_file_dict['end_time'] = (output_offset * 1000) / sample_rate
        output_file_list.append(output_file_dict)
    
    librosa.output.write_wav(output_path, output_audio, sr=sample_rate)
    output_file_pd = pd.DataFrame(output_file_list)
    output_file_pd.to_csv(output_path.split('.')[0] + '.csv', index=False, encoding="utf_8_sig")
    return

def main():
    # mode: [0,1,2]
    # 0: from config file, 'testing' mode audio
    # 1: from folder 
    default_mode = 0
    # default_mode = 1
    default_add_noise_on = False

    parser = argparse.ArgumentParser(description="Prepare XiaoYu Dataset")
    # parser.add_argument('--input_dir', type=str, default='/home/huanyuan/data/speech/kws/weiboyulu/dataset')
    # parser.add_argument('--output_path', type=str, default='/home/huanyuan/data/speech/kws/weiboyulu/straming_dataset/test_unknow_001.wav')
    parser.add_argument('--input_dir', type=str, default="/home/huanyuan/data/speech/kws/xiaoyu_dataset_03022018/XiaoYuDataset_10272020/")
    # parser.add_argument('--output_path', type=str, default="/home/huanyuan/data/speech/kws/xiaoyu_dataset_03022018/straming_dataset/test_unknow_001.wav")
    parser.add_argument('--output_path', type=str, default="/home/huanyuan/data/speech/kws/xiaoyu_dataset_03022018/straming_dataset/test_unknow_002.wav")
    parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", help='config file')
    parser.add_argument('--add_noise_on', type=bool, default=default_add_noise_on)
    parser.add_argument('--mode', type=int, default=default_mode)
    parser.add_argument('--test_duration_seconds', type=int, default=60)
    parser.add_argument('--word_gap_ms', type=int, default=2000)
    args = parser.parse_args()
    straming_dataset_generator(args.input_dir, args.output_path, args.config_file, args.add_noise_on, args.mode, args.test_duration_seconds, args.word_gap_ms)


if __name__ == "__main__":
    main()