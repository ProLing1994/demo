import argparse
import pandas as pd
import sys
from scipy.io import wavfile
import torch.nn.functional as F

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio

from KWS.utils.train_tools import *
from KWS.utils.folder_tools import *
from KWS.dataset.kws.dataset_helper import *
from KWS.impl.pred_pyimpl import kws_load_model, model_predict
from KWS.script.analysis_result.plot_score_line import show_score_line
from KWS.script.analysis_result.cal_fpr_tpr import cal_fpr_tpr
from KWS.impl.recognizer_pyimpl import RecognizeResult, RecognizeCommands, RecognizeCommandsCountNumber, RecognizeCommandsAlign


def generate_results_threshold(args, input_wav, detection_threshold, detection_number_threshold):
    print("Do wave:{}, begin!!!".format(input_wav))

    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # init
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    timeshift_samples = int(sample_rate * cfg.test.timeshift_ms / 1000)
    label_list = cfg.dataset.label.label_list
    num_classes = cfg.dataset.label.num_classes
    positive_label = cfg.dataset.label.positive_label
    positive_label_together = cfg.dataset.label.positive_label_together
    negative_label_together = cfg.dataset.label.negative_label_together
    
    # load label index 
    if positive_label_together and negative_label_together:
        positive_label_together_label_list = cfg.dataset.label.positive_label_together_label
        negative_label_together_label_list = cfg.dataset.label.negative_label_together_label
        label_index = load_label_index(positive_label_together_label_list, negative_label_together_label_list)
    elif positive_label_together:
        positive_label_together_label_list = cfg.dataset.label.positive_label_together_label
        label_index = load_label_index(positive_label_together_label_list, cfg.dataset.label.negative_label)
    elif negative_label_together:
        negative_label_together_label_list = cfg.dataset.label.negative_label_together_label
        label_index = load_label_index(cfg.dataset.label.positive_label, negative_label_together_label_list)
    else:
        label_index = load_label_index(cfg.dataset.label.positive_label, cfg.dataset.label.negative_label)

    if positive_label_together:
        positove_lable_index = label_index[cfg.dataset.label.positive_label_together_label[0]]
    else:
        assert len(positive_label) == 1, "We only support one positive label yet"
        positove_lable_index = label_index[positive_label[0]]

    recognize_element = RecognizeResult()
    if cfg.test.method_mode == 0:
        recognize_commands = RecognizeCommands(
            labels=label_list,
            positove_lable_index = positove_lable_index,
            average_window_duration_ms=cfg.test.average_window_duration_ms,
            detection_threshold=detection_threshold,
            suppression_ms=cfg.test.suppression_ms,
            minimum_count=cfg.test.minimum_count)
    elif cfg.test.method_mode == 1:
        recognize_commands = RecognizeCommandsCountNumber(
            labels=label_list,
            positove_lable_index = positove_lable_index,
            average_window_duration_ms=cfg.test.average_window_duration_ms,
            detection_threshold=detection_threshold,
            detection_number_threshold=detection_number_threshold,
            suppression_ms=cfg.test.suppression_ms,
            minimum_count=cfg.test.minimum_count)
    elif cfg.test.method_mode == 2:
        recognize_commands = RecognizeCommandsAlign(
            labels=label_list,
            positove_lable_index = positove_lable_index,
            average_window_duration_ms=cfg.test.average_window_duration_ms,
            detection_threshold_low=cfg.test.detection_threshold_low,
            detection_threshold_high=cfg.test.detection_threshold_high,
            suppression_ms=cfg.test.suppression_ms,
            minimum_count=cfg.test.minimum_count,
            align_type=cfg.dataset.label.align_type)
    else:
        raise Exception("[ERROR:] Unknow method mode, please check!")
    
    # input dir 
    if args.mode == "0":
        input_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', 
                                    os.path.basename(input_wav).split('.')[0] + '_threshold_{}'.format('_'.join(str(cfg.test.detection_threshold).split('.'))))
    elif args.mode == "2":
        output_subfolder_path = (os.path.dirname(input_wav) + '/').replace(args.input_folder, '')
        input_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', 
                                    args.output_subfolder_name, output_subfolder_path, 
                                    os.path.basename(input_wav).split('.')[0])
    assert os.path.exists(input_dir), input_dir

    # mkdir 
    if args.mode == "0":
        output_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', 
                                    os.path.basename(input_wav).split('.')[0],
                                    'threshold_{}_{}'.format('_'.join(str(detection_threshold).split('.')), ''.join(str(detection_number_threshold).split('.'))))
    elif args.mode == "2":
        output_subfolder_path = (os.path.dirname(input_wav) + '/').replace(args.input_folder, '')
        output_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', 
                                    args.output_subfolder_name, 
                                    output_subfolder_path + '阈值_{}_{}'.format(''.join(str(detection_threshold).split('.')), ''.join(str(detection_number_threshold).split('.'))), 
                                    os.path.basename(input_wav).split('.')[0])
    else:
        raise Exception("[ERROR:] Unknow mode, please check!")
    if os.path.exists(output_dir):    
        return
    else:
        os.makedirs(output_dir)

    # load csv
    csv_pd = pd.read_csv(os.path.join(input_dir, 'original_scores.csv'))

    # load data
    audio_data = librosa.core.load(input_wav, sr=sample_rate)[0]
    assert len(audio_data) > desired_samples, "[ERROR:] Wav is too short! Need more than {} samples but only {} were found".format(
        desired_samples, len(audio_data))

    audio_data_offset = 0
    csv_found_words = []
    for _, row in csv_pd.iterrows():
        print('Done : [{}/{}]'.format(audio_data_offset, len(audio_data)),end='\r')

        # model infer
        output_score = [[0 for i in range(num_classes)]]
        for class_idx in range(num_classes):
            # output_score[0][class_idx] = float(row['score'].split(',')[class_idx])
            output_score[0][class_idx] = float(row['score'].split(',')[class_idx][2:-2])            # '[[0.99727255]]'
        output_score = np.array(output_score)

        # process result
        current_time_ms = int(audio_data_offset * 1000 / sample_rate)
        recognize_commands.process_latest_result(output_score, current_time_ms, recognize_element)

        if recognize_element.is_new_command:
            all_found_words_dict = {}
            all_found_words_dict['label'] = positive_label[0]
            all_found_words_dict['start_time'] = recognize_element.start_time
            all_found_words_dict['end_time'] = recognize_element.end_time + clip_duration_ms
            csv_found_words.append(all_found_words_dict)
            print('Find words: label:{}, start time:{}, end time:{}, response time: {:.2f}s'.format(
                all_found_words_dict['label'], all_found_words_dict['start_time'], all_found_words_dict['end_time'], recognize_element.response_time))

            if bool_write_audio:
                output_path = os.path.join(output_dir, 'label_{}_starttime_{}.wav'.format(all_found_words_dict['label'], all_found_words_dict['start_time']))
                start_time = int(sample_rate * all_found_words_dict['start_time'] / 1000)
                end_time = int(sample_rate * all_found_words_dict['end_time'] / 1000)
                output_wav = audio_data[start_time: end_time]
                audio.save_wav(output_wav, output_path, sample_rate)

        # time ++ 
        audio_data_offset += timeshift_samples

    found_words_pd = pd.DataFrame(csv_found_words)
    found_words_pd.to_csv(os.path.join(output_dir, 'found_words.csv'), index=False)

def main():
    """
    注意：请在运行测试脚本 test_streaming_wav.py 之后运行改脚本，上述脚本已经对音频文件进行测试，获得测试结果。
    本脚本模拟真实音频输入情况，运用不同后处理方法对音频文件进行测试，配置为 --input 中的 config 文件，测试分为以下两种模式：
    1：RecognizeCommands，该脚本会通过滑窗的方式测试每一小段音频数据，计算连续 800ms(27帧) 音频的平均值结果，如果超过预设门限，则认为检测到关键词，否则认定未检测到关键词，最后分别计算假阳性和召回率。
    2：RecognizeCommandsCountNumber，该脚本会通过滑窗的方式测试每一小段音频数据，通过计数超过门限值的个数，判断是否为关键词。
    """
    parser = argparse.ArgumentParser(description='Streamax KWS Testing Engine')
    args = parser.parse_args()

    # mode: [0,1,2]
    # 0: from input_wav_list
    # 1: from csv
    # 2: from folder
    args.mode = "0"    # ["0", "1" ,"2"]

    # mode 0: from input_wav_list
    args.input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/activatebwc_1_5_03312021_validation.wav"]

    # mode 2: from folder
    args.input_folder = ""
    args.output_subfolder_name = ""

    # config file
    args.config_file = "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_english//kws_activatebwc_2_7_tc-resnet14-amba_fbankcpu_kd_09222021/kws_config_activatebwc_api.py"
    
    args.detection_threshold_list = [0.7]
    args.detection_number_threshold_list = [0.3, 0.4]

    if str(args.mode) == "0":
        for detection_threshold in args.detection_threshold_list:
            for detection_number_threshold in args.detection_number_threshold_list:
                for input_wav in args.input_wav_list:
                    generate_results_threshold(args, input_wav, detection_threshold, 
                                                    detection_number_threshold)
    
    if str(args.mode) == "2":
        file_list = get_sub_filepaths_suffix(args.input_folder)
        file_list.sort()

        for detection_threshold in args.detection_threshold_list:
            for detection_number_threshold in args.detection_number_threshold_list:
                for input_wav in file_list:
                    generate_results_threshold(args, input_wav, detection_threshold, 
                                                    detection_number_threshold)

if __name__ == "__main__":
    bool_write_audio = True
    main()