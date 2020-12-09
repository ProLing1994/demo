import argparse
import librosa
import os
import shutil
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *
from dataset.kws.dataset_helper import *
from impl.pred_pyimpl import audio_preprocess
from script.analysis_result.plot_score_line import show_score_line_find_words


def put_together(config_file, input_wav_list, output_folder='error_results'):
    # load configuration file
    cfg = load_cfg_file(config_file)

    # mkdir 
    output_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', output_folder, 'audio')
    if not os.path.exists(output_dir):    
        os.makedirs(output_dir)

    for input_wav in input_wav_list:
        print("Do for wav: {}".format(input_wav))
        input_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', os.path.basename(input_wav).split('.')[0] + '_threshold_{}'.format('_'.join(str(detection_threshold).split('.'))))
        assert os.path.exists(input_dir), "[Error:] Do not find folder: {}, please run test_streaming_wav.py firstly!!!".format(input_dir)

        file_list = os.listdir(input_dir)
        for file_name in file_list:
            if not file_name.endswith('.wav'):
                continue
            
            file_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, os.path.basename(input_dir) + "_" +  file_name)
            print('{} -> {}'.format(file_path, output_path))
            shutil.copy(file_path, output_path)


def cal_features(config_file, output_folder='error_results'):

    # load configuration file
    cfg = load_cfg_file(config_file)

    # init
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    timeshift_samples = int(sample_rate * timeshift_ms / 1000)
    timeshift_times = int((average_window_duration_ms//timeshift_ms + 1) // 2)

    input_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', output_folder, 'audio')
    assert os.path.exists(input_dir), "[Error:] Do not find folder, please put together firstly!!!"

    # mkdir 
    output_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', output_folder, cfg.dataset.preprocess)
    if not os.path.exists(output_dir):    
        os.makedirs(output_dir)


    file_list = os.listdir(input_dir)
    for file_name in file_list:
        print(file_name)
        file_path = os.path.join(input_dir, file_name)

        # load data
        audio_data = librosa.core.load(file_path, sr=sample_rate)[0]
        input_data = audio_data[timeshift_samples*timeshift_times: timeshift_samples*timeshift_times+desired_samples]

        # audio preprocess, load mfcc data
        data = audio_preprocess(cfg, input_data)

        data = data.reshape(-1, 40)
        plot_spectrogram(data.T, os.path.join(output_dir, file_name.split('.')[0] + '.jpg'))


def plot_score(config_file, input_wav_list):
    # load configuration file
    cfg = load_cfg_file(config_file)

    # mkdir 
    output_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', 'error_results')
    if not os.path.exists(output_dir):    
        os.makedirs(output_dir)

    for input_wav in input_wav_list:
        print("Do for wav: {}".format(input_wav))
        input_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', os.path.basename(input_wav).split('.')[0] + '_threshold_{}'.format('_'.join(str(detection_threshold).split('.'))))
        assert os.path.exists(input_dir), "[Error:] Do not find folder, please run test_streaming_wav.py firstly!!!"

        src_csv = input_wav.split('.')[0] + '.csv'
        pst_csv = os.path.join(input_dir, 'mean_scores.csv')
        find_words_csv = os.path.join(input_dir, 'found_words.csv')
        positive_label =  cfg.dataset.label.positive_label[0]

        show_score_line_find_words(src_csv, pst_csv, find_words_csv, output_dir, positive_label)


def main():

    default_config_file = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu3_3_timeshift_spec_on_focal_res15_11032020/test_straming_wav/kws_config_xiaoyu_2.py"

    # statistical_error_results
    # default_input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_3600_001.wav",
    #                         "/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_add_noise_3600_002.wav",
    #                         "/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_43200_003.wav",
    #                         "/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_add_noise_43200_004.wav"]

    # default_input_wav_list = ["/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_001.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_news_cishicike_43200_001.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_001.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_zhongdongwangshi_7200_001.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_yeshimiwen_43200_001.wav"]

    # difficult sample mining
    # default_input_wav_list = ["/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_baijiajiangtan_21600_noused_001.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_jinpingmei_21600_noused_001.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_001.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_002.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_003.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_zhongdongwangshi_21600_noused_001.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_001.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_002.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_003.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_xingetuijian_21600_noused_001.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_xingetuijian_21600_noused_002.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_news_cishicike_21600_noused_001.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_news_cishicike_21600_noused_002.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_news_cishicike_21600_noused_003.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_001.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_002.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_003.wav"]

    default_input_wav_list = ["/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_004.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_005.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_006.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_007.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_008.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_009.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_004.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_005.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_006.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_007.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_008.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_009.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_004.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_005.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_006.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_007.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_008.wav",
                            "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_009.wav"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=default_config_file, nargs='?', help='config file')
    parser.add_argument('--input_wav_list', type=str, default=default_input_wav_list)
    args = parser.parse_args()

    # statistical_error_results
    # put_together(args.config_file, args.input_wav_list)
    # cal_features(args.config_file)
    # plot_score(args.config_file, args.input_wav_list)

    # difficult sample mining
    # put_together(args.config_file, args.input_wav_list, 'difficult_mining')
    # cal_features(args.config_file, 'difficult_mining')

if __name__ == '__main__':
    timeshift_ms = 30
    average_window_duration_ms = 800
    detection_threshold = 0.95
    main()