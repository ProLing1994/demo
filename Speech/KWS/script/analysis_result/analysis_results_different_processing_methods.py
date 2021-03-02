import argparse
import collections
import multiprocessing 
import pandas as pd
import pickle
import sys
import time 
import torch.nn.functional as F

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *
from dataset.kws.dataset_helper import *
from impl.pred_pyimpl import kws_load_model, model_predict
from script.analysis_result.plot_score_line import show_score_line
from script.analysis_result.cal_fpr_tpr import cal_fpr_tpr
from impl.recognizer_pyimpl import RecognizeResult, RecognizeCommands, RecognizeCommandsCountNumber

def generate_results_threshold(input_wav, config_file, detection_threshold, detection_number_threshold):
    print("Do wave:{}, begin!!!".format(input_wav))

    # load configuration file
    cfg = load_cfg_file(config_file)

    # init
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    timeshift_samples = int(sample_rate * cfg.test.timeshift_ms / 1000)
    label_list = cfg.dataset.label.label_list
    num_classes = cfg.dataset.label.num_classes
    positive_label = cfg.dataset.label.positive_label
    assert len(positive_label) == 1, "We only support one positive label yet"
    
    # load label index 
    label_index = load_label_index(cfg.dataset.label.positive_label, cfg.dataset.label.negative_label)

    recognize_element = RecognizeResult()
    if method_mode == 0:
        recognize_commands = RecognizeCommands(
            labels=label_list,
            positove_lable_index = label_index[positive_label[0]],
            average_window_duration_ms=cfg.test.average_window_duration_ms,
            detection_threshold=detection_threshold,
            suppression_ms=cfg.test.suppression_ms,
            minimum_count=cfg.test.minimum_count)
    elif method_mode == 1:
        recognize_commands = RecognizeCommandsCountNumber(
            labels=label_list,
            positove_lable_index = label_index[positive_label[0]],
            average_window_duration_ms=cfg.test.average_window_duration_ms,
            detection_threshold=detection_threshold,
            detection_number_threshold=detection_number_threshold,
            suppression_ms=cfg.test.suppression_ms,
            minimum_count=cfg.test.minimum_count)
    else:
        raise Exception("[ERROR:] Unknow method mode, please check!")
    
    # input dir 
    input_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', os.path.basename(input_wav).split('.')[0] + '_threshold_{}'.format("_".join(str(default_threshold).split('.'))))
    assert os.path.exists(input_dir), input_dir

    # mkdir 
    output_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', os.path.basename(input_wav).split('.')[0] + '_thresholds', 
                                'method_mode_{}'.format(method_mode),
                                'threshold_{}_{}'.format('_'.join(str(detection_threshold).split('.')), '_'.join(str(detection_number_threshold).split('.'))))
    if not os.path.exists(output_dir):    
        os.makedirs(output_dir)

    # load csv
    csv_pd = pd.read_csv(os.path.join(input_dir, 'original_scores.csv'))

    # load data
    audio_data = librosa.core.load(input_wav, sr=sample_rate)[0]
    assert len(audio_data) > desired_samples, "[ERROR:] Wav is too short! Need more than {} samples but only {} were found".format(
        desired_samples, len(audio_data))

    audio_data_offset = 0
    all_found_words = []
    for _, row in csv_pd.iterrows():
        print('Done : [{}/{}]'.format(audio_data_offset, len(audio_data)),end='\r')

        # model infer result
        score = float(row['score'].split(',')[-1])
        output_score = [[0 for i in range(num_classes)]]
        output_score[0][label_index[positive_label[0]]] = score
        output_score = np.array(output_score)

        # process result
        current_time_ms = int(audio_data_offset * 1000 / sample_rate)
        recognize_commands.process_latest_result(output_score, current_time_ms, recognize_element)

        if recognize_element.is_new_command:
            all_found_words_dict = {}
            all_found_words_dict['label'] = positive_label[0]
            all_found_words_dict['start_time'] = recognize_element.start_time
            all_found_words_dict['end_time'] = recognize_element.end_time + clip_duration_ms
            all_found_words.append(all_found_words_dict)
            print('Find words: label:{}, start time:{}, end time:{}, response time: {:.2f}s'.format(
                all_found_words_dict['label'], all_found_words_dict['start_time'], all_found_words_dict['end_time'], recognize_element.response_time))

            if bool_write_audio:
                output_path = os.path.join(output_dir, 'label_{}_starttime_{}.wav'.format(all_found_words_dict['label'], all_found_words_dict['start_time']))
                start_time = int(sample_rate * all_found_words_dict['start_time'] / 1000)
                end_time = int(sample_rate * all_found_words_dict['end_time'] / 1000)
                output_wav = audio_data[start_time: end_time]
                librosa.output.write_wav(output_path, output_wav, sr=sample_rate)

        # time ++ 
        audio_data_offset += timeshift_samples

    found_words_pd = pd.DataFrame(all_found_words)
    found_words_pd.to_csv(os.path.join(output_dir, 'found_words.csv'), index=False)

def main():
    """
    注意：请在运行测试脚本 test_streaming_wav.py 之后运行改脚本，上述脚本已经对音频文件进行测试，获得测试结果。
    本脚本模拟真实音频输入情况，运用不同后处理方法对音频文件进行测试，配置为 --input 中的 config 文件，测试分为以下两种模式：
    1：RecognizeCommands，该脚本会通过滑窗的方式测试每一小段音频数据，计算连续 800ms(27帧) 音频的平均值结果，如果超过预设门限，则认为检测到关键词，否则认定未检测到关键词，最后分别计算假阳性和召回率。
    2：RecognizeCommandsCountNumber，该脚本会通过滑窗的方式测试每一小段音频数据，通过计数超过门限值的个数，判断是否为关键词。
    """
    # test
    # default_input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/xiaorui_12162020_validation_3600_001.wav",
    #                             "/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_3600_001.wav"]

    default_input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_43200_003.wav",
                                "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_news_cishicike_43200_001.wav",
                                "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_001.wav",
                                "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_001.wav",
                                "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_yeshimiwen_43200_001.wav",
                                "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_zhongdongwangshi_7200_001.wav",
                                "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_xingetuijian_21600_001.wav"]

    default_config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui.py"
    default_detection_threshold_list = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    default_detection_number_threshold_list = [0.5, 0.75, 0.9]

    parser = argparse.ArgumentParser(description='Streamax KWS Testing Engine')
    parser.add_argument('--input_wav_list', type=str,
                        default=default_input_wav_list)
    parser.add_argument('--config_file', type=str,
                        default=default_config_file)
    parser.add_argument('--detection_threshold_list',
                        type=int, default=default_detection_threshold_list)
    parser.add_argument('--detection_number_threshold_list',
                        type=int, default=default_detection_number_threshold_list)
    args = parser.parse_args()

    for detection_threshold in args.detection_threshold_list:
        for detection_number_threshold in args.detection_number_threshold_list:
            for input_wav in args.input_wav_list:
                generate_results_threshold(input_wav, args.config_file, detection_threshold, 
                                                detection_number_threshold)

if __name__ == "__main__":
    method_mode = 1             # [0: RecognizeCommands, 1: RecognizeCommandsCountNumber]
    default_threshold = 0.8     # [0.8, 0.95]
    bool_write_audio = True
    main()