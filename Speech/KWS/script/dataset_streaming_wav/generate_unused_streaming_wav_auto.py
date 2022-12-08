import argparse
import librosa
import numpy as np
import os 
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio

from KWS.config.kws import hparams
from KWS.utils.train_tools import load_cfg_file
from KWS.script.dataset_streaming_wav.generate_streaming_wav import mix_in_audio_sample


def straming_dataset_generator(input_dir, output_format, nosed_csv, config_file, mode, test_duration_seconds, word_gap_ms):
    # load configuration file
    cfg = load_cfg_file(config_file)

    # init
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    output_audio_sample_count = sample_rate * test_duration_seconds
    word_gap_samples = int((word_gap_ms * sample_rate) / 1000)
    start_idx = 0
    file_idx = 1

    # mkdir
    if not os.path.exists(os.path.dirname(output_format)):
        os.makedirs(os.path.dirname(output_format))

    # mode
    audio_list = [] # {'file': ..., 'label': ...}
    if mode == 2:
        print("Generator Straming Dataset From Unused Csv: {}".format(nosed_csv))
        data_pd = pd.read_csv(nosed_csv)
        data_pd.sort_values(by='file', inplace=True)
        for _, row in data_pd.iterrows():
            audio_dict = {}
            audio_dict['file'] = row['file']
            audio_dict['label'] = hparams.UNKNOWN_WORD_LABEL
            audio_list.append(audio_dict)
    else:
        pass
    
    while(start_idx < len(audio_list)):
        output_audio = np.zeros((output_audio_sample_count,), dtype=np.float32)

        # Mix the audio into the main track, noting their labels and positions.
        output_file_list = [] # {'file': ..., 'label': ..., 'start_time': ..., 'end_time': ...}
        output_offset = 0
        while(output_offset < output_audio_sample_count):

            print('Mix audio, Done : [{}/{}]'.format(output_offset, output_audio_sample_count), end='\r')
            output_offset += word_gap_samples + np.random.randint(word_gap_samples)
            output_offset_ms = (output_offset * 1000) / sample_rate

            if output_offset >= output_audio_sample_count:
                break

            if mode == 2:
                data_index = start_idx
                if data_index >= len(audio_list):
                    break

                found_data = audio_list[data_index]['file']
                found_label = audio_list[data_index]['label']
                start_idx += 1
            else:
                pass 
            
            # load data
            if found_label == hparams.SILENCE_LABEL:
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

        if output_offset < output_audio_sample_count:
            output_audio = output_audio[:output_offset]

        output_path = output_format + '{:0>3d}.wav'.format(file_idx)
        file_idx += 1
        audio.save_wav(output_audio.copy(), output_path, sample_rate)
        output_file_pd = pd.DataFrame(output_file_list)
        output_file_pd.to_csv(output_path.split('.')[0] + '.csv', index=False, encoding="utf_8_sig")
        print("Wav Done: {}".format(output_path))
    return

def main():
    # mode: [0,1,2]
    # 0: from config file
    # 1: from folder 
    # 2: from unused csv 
    # default_mode = 0
    # default_mode = 1 
    default_mode = 2
    
    # only for mode==2, from unused csv 
    # default_input_dir = '/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/baijiajiangtan'
    # default_output_format = '/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_baijiajiangtan_21600_noused_'
    # default_nosed_csv = "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/QingTingFM_history_baijiajiangtan_21600_noused.csv"

    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/jinpingmei/"
    # default_output_format = "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_jinpingmei_21600_noused_"
    # default_nosed_csv = "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/QingTingFM_history_jinpingmei_7200_noused.csv"

    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/yeshimiwen/"
    # default_output_format = "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_"
    # default_nosed_csv = "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/QingTingFM_history_yeshimiwen_43200_noused.csv"

    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/zhongdongwangshi/"
    # default_output_format = "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_zhongdongwangshi_21600_noused_"
    # default_nosed_csv = "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/QingTingFM_history_zhongdongwangshi_7200_noused.csv"
    
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/news/cishicike/"
    # default_output_format = "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_news_cishicike_21600_noused_"
    # default_nosed_csv  ="/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/QingTingFM_news_cishicike_43200_noused.csv"
    
    # default_input_dir =   "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/music_station/qingtingkongzhongyinyuebang/"
    # default_output_format = "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_"
    # default_nosed_csv = "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_noused.csv"

    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/music/xingetuijian/"
    # default_output_format = '/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_xingetuijian_21600_noused_'
    # default_nosed_csv = "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/QingTingFM_music_xingetuijian_21600_noused.csv"

    default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/novel/douluodalu/"
    default_output_format = '/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_'
    default_nosed_csv = "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/QingTingFM_novel_douluodalu_43200_noused.csv"
    
    default_config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py"

    parser = argparse.ArgumentParser(description="Prepare XiaoYu Dataset")
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--output_format', type=str, default=default_output_format)
    parser.add_argument('--nosed_csv', type=str, default=default_nosed_csv)
    parser.add_argument('--config_file', type=str, default=default_config_file, help='config file')
    parser.add_argument('--mode', type=int, default=default_mode)
    # parser.add_argument('--test_duration_seconds', type=int, default=43200) # 12 hours
    parser.add_argument('--test_duration_seconds', type=int, default=21600) # 6 hours
    # parser.add_argument('--test_duration_seconds', type=int, default=7200) # 2 hours
    # parser.add_argument('--word_gap_ms', type=int, default=3000)
    parser.add_argument('--word_gap_ms', type=int, default=1000)
    args = parser.parse_args()
    
    straming_dataset_generator(args.input_dir, args.output_format, args.nosed_csv, args.config_file, args.mode, args.test_duration_seconds, args.word_gap_ms)


if __name__ == "__main__":
    main()