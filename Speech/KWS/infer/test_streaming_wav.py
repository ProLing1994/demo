import argparse
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
from impl.recognizer_pyimpl import RecognizeResult, RecognizeCommands, RecognizeCommandsCountNumber, RecognizeCommandsAlign
from script.analysis_result.plot_score_line import show_score_line
from script.analysis_result.cal_fpr_tpr import cal_fpr_tpr


def test(input_wav, args):
# def test(in_args):
#     input_wav = in_args[0]
#     args = in_args[1]

    print("Do wave:{}, begin!!!".format(input_wav))

    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # init
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    timeshift_samples = int(sample_rate * cfg.test.timeshift_ms / 1000)
    label_list = cfg.dataset.label.label_list
    positive_label = cfg.dataset.label.positive_label

    # load label index 
    label_index = load_label_index(cfg.dataset.label.positive_label, cfg.dataset.label.negative_label)

    recognize_element = RecognizeResult()
    if cfg.test.method_mode == 0:
        recognize_commands = RecognizeCommands(
            labels=label_list,
            positove_lable_index = label_index[positive_label[0]],
            average_window_duration_ms=cfg.test.average_window_duration_ms,
            detection_threshold=cfg.test.detection_threshold,
            suppression_ms=cfg.test.suppression_ms,
            minimum_count=cfg.test.minimum_count)
    elif cfg.test.method_mode == 1:
        recognize_commands = RecognizeCommandsCountNumber(
            labels=label_list,
            positove_lable_index = label_index[positive_label[0]],
            average_window_duration_ms=cfg.test.average_window_duration_ms,
            detection_threshold=cfg.test.detection_threshold,
            detection_number_threshold=cfg.test.detection_number_threshold,
            suppression_ms=cfg.test.suppression_ms,
            minimum_count=cfg.test.minimum_count)
    elif cfg.test.method_mode == 2:
        recognize_commands = RecognizeCommandsAlign(
            labels=label_list,
            positove_lable_index = label_index[positive_label[0]],
            average_window_duration_ms=cfg.test.average_window_duration_ms,
            detection_threshold_low=cfg.test.detection_threshold_low,
            detection_threshold_high=cfg.test.detection_threshold_high,
            suppression_ms=cfg.test.suppression_ms,
            minimum_count=cfg.test.minimum_count,
            align_type=cfg.dataset.label.align_type)
    else:
        raise Exception("[ERROR:] Unknow method mode, please check!")

    # mkdir 
    if args.mode == "0":
        output_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', 
                                    os.path.basename(input_wav).split('.')[0] + '_threshold_{}'.format('_'.join(str(cfg.test.detection_threshold).split('.'))))
    elif args.mode == "1":
        output_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', 
                                    os.path.basename(args.csv_path).split('.')[0], args.type, 'bool_noise_reduction_' + str(args.bool_noise_reduction),
                                    os.path.basename(input_wav).split('.')[0] + '_threshold_{}'.format('_'.join(str(cfg.test.detection_threshold).split('.'))))
    else:
        raise Exception("[ERROR:] Unknow mode, please check!")
    if not os.path.exists(output_dir):    
        os.makedirs(output_dir)
    
    # load model
    model = kws_load_model(cfg.general.save_dir, int(cfg.general.gpu_ids), cfg.test.model_epoch)
    net = model['prediction']['net']
    net.eval()

    # load data
    audio_data = librosa.core.load(input_wav, sr=sample_rate)[0]

    # alignment data, 模拟长时语音
    if len(audio_data) < 6 * desired_samples:
        data_length = len(audio_data)
        audio_data = np.pad(audio_data, (max(0, (6 * desired_samples - data_length)//2), 0), "constant")
        audio_data = np.pad(audio_data, (0, max(0, (6 * desired_samples - data_length + 1)//2)), "constant")
    assert len(audio_data) >= 6 * desired_samples, "[ERROR:] Wav is too short! Need more than {} samples but only {} were found".format(
        desired_samples, len(audio_data))

    audio_data_offset = 0
    original_scores = [] 
    final_scores = [] 
    all_found_words = []

    # record time
    start = time.perf_counter()
    model_predict_time_list = []

    while(audio_data_offset < len(audio_data)):
        print('Done : [{}/{}]'.format(audio_data_offset, len(audio_data)),end='\r')

        # input data
        input_start = audio_data_offset
        input_end = audio_data_offset + desired_samples
        input_data = audio_data[input_start: input_end]
        if len(input_data) != desired_samples:
            break

        # model infer
        model_predict_start_time = time.perf_counter()
        output_score = model_predict(cfg, net, input_data)
        model_predict_end_time = time.perf_counter()
        model_predict_time_list.append(model_predict_end_time - model_predict_start_time)

        # process result
        current_time_ms = int(input_start * 1000 / sample_rate)
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

        original_scores.append({'start_time':current_time_ms, 'score':",".join([str(output_score[0][idx]) for idx in range(output_score.shape[1])])})
        final_scores.append({'start_time':current_time_ms, 'score':recognize_element.score})

        # time ++ 
        audio_data_offset += timeshift_samples

    # record time
    end = time.perf_counter()
    print('Running time: {:.2f} Seconds'.format(end - start))
    print('Model predict numbers: {}, average time: {:.3f}s'.format(len(model_predict_time_list), np.array(model_predict_time_list).sum() / len(model_predict_time_list)))

    found_words_pd = pd.DataFrame(all_found_words)
    found_words_pd.to_csv(os.path.join(output_dir, 'found_words.csv'), index=False)
    original_scores_pd = pd.DataFrame(original_scores)
    original_scores_pd.to_csv(os.path.join(output_dir, 'original_scores.csv'), index=False)
    final_scores_pd = pd.DataFrame(final_scores)
    final_scores_pd.to_csv(os.path.join(output_dir, 'final_scores.csv'), index=False)
    
    # show result
    # [TO DO]：画图在多进程中会挂掉，目前在生成结果之后，再单独画图
    # show_score_line(input_wav.split('.')[0] + '.csv', os.path.join(output_dir, 'original_scores.csv'), positive_label[0])
    # show_score_line(input_wav.split('.')[0] + '.csv', os.path.join(output_dir, 'mean_scores.csv'), positive_label[0])

    # cal_fpr_tpr(input_wav.split('.')[0] + '.csv', os.path.join(output_dir, 'found_words.csv'),  positive_label[0], bool_write_audio)

    print("Do wave:{}, Done!!!".format(input_wav))


def main():
    """
    本脚本模拟真实音频输入情况，对音频文件进行测试，配置为 --input 中的 config 文件。
    测试过程：该脚本会通过滑窗的方式测试每一小段音频数据，计算连续 800ms(27帧)/2000ms(41帧) 音频的平均值结果，如果超过预设门限，则认为检测到关键词，否则认定未检测到关键词，最后分别计算假阳性和召回率。
    """   
    # mode: [0,1,2]
    # 0: from input_wav_list
    # 1: from csv
    default_mode = "0"    # ["0", "1"]

    # mode 0: from input_wav_list
    # test
    # default_input_wav_list = ["/home/huanyuan/share/audio_data/RM_KWS_XIAORUI_xiaorui_S001M1D00T001.wav",
    #                         "/home/huanyuan/share/audio_data/RM_KWS_XIAORUI_xiaorui_S001M1D00T002.wav"]

    # xiaoyu
    # default_input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/xiaoyu_12042020_training_60_001.wav",
    #                         "/mnt/huanyuan/model/test_straming_wav/xiaoyu_12042020_validation_60_001.wav",
    #                         "/mnt/huanyuan/model/test_straming_wav/xiaoyu_12042020_testing_60_001.wav"]
    # default_input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/xiaoyu_12042020_testing_3600_001.wav",
    #                         "/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_3600_001.wav"]

    # xiaorui
    default_input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/xiaorui_12162020_training_60_001.wav",
                                "/mnt/huanyuan/model/test_straming_wav/xiaorui_12162020_validation_60_001.wav"]
    # default_input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/xiaorui_12162020_validation_3600_001.wav",
    #                         "/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_3600_001.wav"]

    # xiaole
    # default_input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/xiaole_11252020_training_60_001.wav",
    #                             "/mnt/huanyuan/model/test_straming_wav/xiaole_11252020_validation_60_001.wav",
    #                             "/mnt/huanyuan/model/test_straming_wav/xiaole_11252020_testing_60_001.wav"]
    # default_input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/xiaole_11252020_testing_3600_001.wav",
    #                         "/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_3600_001.wav"]

    # pretrain
    # default_input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/pretrain_12102020_training_60_001.wav",
    #                         "/mnt/huanyuan/model/test_straming_wav/pretrain_12102020_validation_60_001.wav"]
    # default_input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/pretrain_12102020_validation_3600_001.wav",
    #                         "/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_3600_001.wav"]
    # default_input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/pretrain_1_1_12212020_training_60_001.wav",
    #                         "/mnt/huanyuan/model/test_straming_wav/pretrain_1_1_12212020_validation_60_001.wav"]
    # default_input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/pretrain_1_1_12212020_validation_3600_001.wav",
    #                         "/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_3600_001.wav"]

    # nagetive test
    # default_input_wav_list = ["/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_43200_003.wav",
    #                             "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_news_cishicike_43200_001.wav",
    #                             "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_001.wav",
    #                             "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_001.wav",
    #                             "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_yeshimiwen_43200_001.wav",
    #                             "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_zhongdongwangshi_7200_001.wav",
    #                             "/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_xingetuijian_21600_001.wav"]

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
    # default_input_wav_list = ["/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_004.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_005.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_006.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_007.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_008.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_history_yeshimiwen_21600_noused_009.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_004.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_005.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_006.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_007.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_008.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_21600_noused_009.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_004.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_005.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_006.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_007.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_008.wav",
    #                         "/mnt/huanyuan/data/speech/Negative_sample/noused_in_test_straming_wav/noused_straming_wav/QingTingFM_novel_douluodalu_21600_noused_009.wav"]
    # default_input_wav_list = ["/mnt/huanyuan/data/speech/Negative_sample/CollectVoice/Jabra_510/Jabra_510_background_noise_001.wav",
    #                             "/mnt/huanyuan/data/speech/Negative_sample/CollectVoice/Jabra_510/Jabra_510_background_noise_002.wav",
    #                             "/mnt/huanyuan/data/speech/Negative_sample/CollectVoice/Jabra_510/Jabra_510_background_noise_003.wav"]

    # mode 1: from csv
    default_csv_path = "/mnt/huanyuan/data/speech/Real_vehicle_sample/20201218/Real_vehicle_sample_20201218.csv"
    default_type = 'normal_driving'                 # ['normal_driving', 'idling_driving']
    default_bool_noise_reduction = False            # [False, True]

    # config file
    # default_config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py"
    default_config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui.py"
    # default_config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaole.py"
    # default_config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_2_label_xiaoyu.py"
    # default_config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_pretrain.py"

    # align config file
    # default_config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_align_xiaoyu.py"
    # default_config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_align_pretrain.py"
    # default_config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_align_xiaorui.py"

    parser = argparse.ArgumentParser(description='Streamax KWS Testing Engine')
    parser.add_argument('--mode', type=str, default=default_mode)
    parser.add_argument('--input_wav_list', type=str, default=default_input_wav_list)
    parser.add_argument('--csv_path', type=str, default=default_csv_path)
    parser.add_argument('--type', type=str, default=default_type)
    parser.add_argument('--bool_noise_reduction', action='store_true', default=default_bool_noise_reduction)
    parser.add_argument('--config_file', type=str, default=default_config_file)
    args = parser.parse_args()

    if str(args.mode) == "0":
        # in_params = []
        # for input_wav in args.input_wav_list:
        #     in_args = [input_wav, args]
        #     in_params.append(in_args)

        # p = multiprocessing.Pool(3)
        # out = p.map(test, in_params)
        # p.close()
        # p.join()

        for input_wav in args.input_wav_list:
            test(input_wav, args)
    elif str(args.mode) == "1":
        dataset_pd = pd.read_csv(args.csv_path)
        dataset_pd = dataset_pd[dataset_pd["type"] == args.type]
        dataset_pd = dataset_pd[dataset_pd["bool_noise_reduction"] == args.bool_noise_reduction]

        in_params = []
        for _, row in dataset_pd.iterrows():
            in_args = [row['path'], args]
            in_params.append(in_args)

        p = multiprocessing.Pool(3)
        out = p.map(test, in_params)
        p.close()
        p.join()

        # for _, row in dataset_pd.iterrows():
        #     test(row['path'], args)
    else:
        raise Exception("[ERROR:] Unknow mode, please check!")


if __name__ == "__main__":
    bool_write_audio = True
    main()