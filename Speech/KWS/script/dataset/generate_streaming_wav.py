import argparse
import librosa
import numpy as np
import os 
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import load_cfg_file
from dataset.kws.dataset_helper import *
from impl.pred_pyimpl import load_background_noise, dataset_add_noise


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


def straming_dataset_generator(input_dir, output_path, nosed_csv, config_file, add_noise_on, mode, audio_mode, test_duration_seconds, word_gap_ms):
    # load configuration file
    cfg = load_cfg_file(config_file)

    # init
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    output_audio_sample_count = sample_rate * test_duration_seconds
    output_audio = np.zeros((output_audio_sample_count,), dtype=np.float32)
    word_gap_samples = int((word_gap_ms * sample_rate) / 1000)

    # mkdir
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # mode
    audio_list = [] # {'file': ..., 'label': ...}
    if mode == 0:
        print("Generator Straming Dataset From Config File")
        data_pd = pd.read_csv(cfg.general.data_csv_path)
        data_pd = data_pd[data_pd['mode'] == audio_mode]
        for _, row in data_pd.iterrows():
            audio_dict = {}
            audio_dict['file'] = row['file']
            audio_dict['label'] = row['label']
            audio_list.append(audio_dict)
    elif mode == 1:
        print("Generator Straming Dataset From Folder")
        file_list = os.listdir(input_dir)
        for file in file_list:
            audio_dict = {}
            audio_dict['file'] = os.path.join(input_dir, file)
            audio_dict['label'] = UNKNOWN_WORD_LABEL
            audio_list.append(audio_dict)
    elif mode == 2:
        print("Generator Straming Dataset From Unused Csv")
        # init 
        start_idx = 0

        data_pd = pd.read_csv(nosed_csv)
        data_pd.sort_values(by='file', inplace=True)
        for _, row in data_pd.iterrows():
            audio_dict = {}
            audio_dict['file'] = row['file']
            audio_dict['label'] = UNKNOWN_WORD_LABEL
            audio_list.append(audio_dict)
    else:
        pass

    # Mix the background audio into the main track.
    if add_noise_on:

        background_data = load_background_noise(cfg)
        output_offset = 0
        while(output_offset < output_audio_sample_count):
            print('Mix background audio, Done : [{}/{}]'.format(output_offset, output_audio_sample_count), end='\r')

            if output_offset >= output_audio_sample_count:
                break
        
            # load data
            audio_data = np.zeros(desired_samples, dtype=np.float32)
            audio_data = dataset_add_noise(cfg, audio_data, background_data)

            audio_length = len(audio_data)
            mix_in_audio_sample(output_audio, output_offset, audio_data, 0, audio_length, 1.0, 500, 500)
            output_offset += audio_length

    # Mix the audio into the main track, noting their labels and positions.
    print()
    output_file_list = [] # {'file': ..., 'label': ..., 'start_time': ..., 'end_time': ...}
    output_offset = 0
    while(output_offset < output_audio_sample_count):
        print('Mix audio, Done : [{}/{}]'.format(output_offset, output_audio_sample_count), end='\r')

        output_offset += word_gap_samples + np.random.randint(word_gap_samples)
        output_offset_ms = (output_offset * 1000) / sample_rate

        if output_offset >= output_audio_sample_count :
            break
        if mode == 0:
            data_index = np.random.randint(len(audio_list))
            found_data = audio_list[data_index]['file']
            found_label = audio_list[data_index]['label']
        elif mode == 1:
            data_index = np.random.randint(len(audio_list))
            found_data = audio_list[data_index]['file']
            found_label = audio_list[data_index]['label']
        elif mode == 2:
            data_index = start_idx
            if data_index >= len(audio_list):
                break

            found_data = audio_list[data_index]['file']
            found_label = audio_list[data_index]['label']
            start_idx += 1
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
        output_file_dict['label'] = found_label
        output_file_dict['start_time'] = output_offset_ms
        output_file_dict['end_time'] = (output_offset * 1000) / sample_rate
        output_file_list.append(output_file_dict)
    
    librosa.output.write_wav(output_path, output_audio, sr=sample_rate)
    output_file_pd = pd.DataFrame(output_file_list)
    output_file_pd.to_csv(output_path.split('.')[0] + '.csv', index=False, encoding="utf_8_sig")
    return

def main():
    # mode: [0,1,2]
    # 0: from config file
    # 1: from folder 
    # 2: from unused csv 
    # default_mode = 0
    # default_mode = 1 
    default_mode = 2

    default_add_noise_on = False
    # default_add_noise_on = True

    # only for mode==0, support for ['training','validation','testing']
    default_audio_mode = 'testing'

    # only for mode==1, from folder
    # default_input_dir = '/mnt/huanyuan/data/speech/kws/weiboyulu/dataset'
    # default_output_path_list = ['/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_43200_003.wav']
    # default_output_path_list = ['/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_add_noise_43200_004.wav']
    # default_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_03022018/XiaoYuDataset_10272020/"
    # default_output_path_list = ["/mnt/huanyuan/model/test_straming_wav/xiaoyu_03022018_testing_3600_001.wav"]
    # default_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_10292020/XiaoYuDataset_10292020/"
    # default_output_path_list = ["/mnt/huanyuan/model/test_straming_wav/xiaoyu_10292020_testing_3600_001.wav"]

    # default_input_dir = '/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/baijiajiangtan'
    # default_output_path_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_baijiajiangtan_21600_001.wav']
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/jinpingmei/"
    # default_output_path_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_jinpingmei_7200_001.wav']
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/yeshimiwen/"
    # default_output_path_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_yeshimiwen_43200_001.wav',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_yeshimiwen_43200_002.wav',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_yeshimiwen_43200_003.wav',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_yeshimiwen_43200_004.wav',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_yeshimiwen_43200_005.wav']
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/zhongdongwangshi/"
    # default_output_path_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_zhongdongwangshi_7200_001.wav']
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/news/cishicike/"
    # default_output_path_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_news_cishicike_43200_001.wav',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_news_cishicike_43200_002.wav',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_news_cishicike_43200_003.wav']
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/music_station/qingtingkongzhongyinyuebang/"
    # default_output_path_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_008.wav',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_009.wav',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_010.wav',]
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/novel/douluodalu/"
    # default_output_path_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_006.wav',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_007.wav',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_008.wav',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_009.wav',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_010.wav',]
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/music/xingetuijian/"
    # default_output_path_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_xingetuijian_21600_001.wav']
    
    # only for mode==2, from unused csv 
    default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/music/xingetuijian/"
    default_output_path_list = ['/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_xingetuijian_21600_noused_001.wav']
    default_nosed_csv = "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/QingTingFM_music_xingetuijian_21600_noused.csv"
    
    # default_config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py"
    default_config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu_2.py"

    parser = argparse.ArgumentParser(description="Prepare XiaoYu Dataset")
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--output_path_list', type=str, default=default_output_path_list)
    parser.add_argument('--nosed_csv', type=str, default=default_nosed_csv)
    parser.add_argument('--config_file', type=str, default=default_config_file, help='config file')
    parser.add_argument('--add_noise_on', type=bool, default=default_add_noise_on)
    parser.add_argument('--mode', type=int, default=default_mode)
    parser.add_argument('--audio_mode', type=str, default=default_audio_mode)
    # parser.add_argument('--test_duration_seconds', type=int, default=43200) # 12 hours
    parser.add_argument('--test_duration_seconds', type=int, default=21600) # 6 hours
    # parser.add_argument('--test_duration_seconds', type=int, default=7200) # 2 hours
    # parser.add_argument('--word_gap_ms', type=int, default=3000)
    parser.add_argument('--word_gap_ms', type=int, default=1000)
    args = parser.parse_args()
    
    for output_path in args.output_path_list:
        print("Do wave:{}, begin!!!".format(output_path))
        straming_dataset_generator(args.input_dir, output_path, args.nosed_csv, args.config_file, args.add_noise_on, args.mode, args.audio_mode, args.test_duration_seconds, args.word_gap_ms)
        print("Do wave:{}, done!!!".format(output_path))


if __name__ == "__main__":
    main()