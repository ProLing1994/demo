import argparse
import pandas as pd
import sys

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.utils.train_tools import *

from KWS.config.kws import hparams
from KWS.dataset.kws import dataset_augmentation
from KWS.dataset.kws.dataset_helper import load_label_index
from KWS.impl.pred_pyimpl import model_predict
from KWS.impl.recognizer_pyimpl import DoubleEdgeDetecting
from KWS.utils.lmdb_tools import load_background_noise_lmdb, load_lmdb_env, read_audio_lmdb


def longterm_audio_align_post_processing(cfg, score_list, result_mode, timeshift_ms, average_window_duration_ms):
    """
    后处理操作，基于帧对齐模式
    """
    final_scores = []
    final_scores.append(0.0)
    if result_mode == 'double_edge_triggered_detecting':
        average_window_length = 1 + average_window_duration_ms // timeshift_ms

        if len(score_list) > average_window_length:
            windows_number = 1 + len(score_list) - average_window_length
            # Calculate the average score across all the results in the window.
            scores_list = []

            for windows_idx in range(windows_number):
                score_list_window = score_list[windows_idx: windows_idx + average_window_length]
                
                if cfg.dataset.label.align_type == "transform":
                    # 后处理方法：检测结果：unknow、小鱼、鱼小，检测边缘：小鱼、鱼小、小鱼，故将 3 维检测结果拼接为 4 维
                    score_list_window = np.array(score_list_window)
                    score_list_window = np.concatenate((score_list_window, score_list_window[:, 1].reshape(score_list_window.shape[0], 1)), axis=1)
                    score = DoubleEdgeDetecting.compute_conf(score_list_window, word_num=3)
                    scores_list.append(score)
                
                elif cfg.dataset.label.align_type == "word":
                    # 后处理方法：检测结果：unknow、小、鱼，检测边缘：小、鱼、小、鱼，故将 3 维检测结果拼接为 5 维
                    score_list_window = np.array(score_list_window)
                    score_list_window = np.concatenate((score_list_window, score_list_window[:, 1:3].reshape(score_list_window.shape[0], 2)), axis=1)
                    score = DoubleEdgeDetecting.compute_conf(score_list_window, word_num=4)
                    scores_list.append(score)
                else:
                    raise Exception("[ERROR] Unknow align_type: {}, please check!".fomrat(cfg.dataset.label.align_type))

            # Sort the averaged results.
            scores_list.sort()
            final_scores.append(1 - scores_list[-1])
            final_scores.append(scores_list[-1])

    else:
        raise Exception("[ERROR:] Unknow result mode, please check!")

    pred = np.argmax(final_scores)
    return pred, final_scores


def longterm_audio_post_processing(cfg, score_list, audio_label_idx, result_mode, timeshift_ms, average_window_duration_ms):
    """
    后处理操作
    """
    num_classes = cfg.dataset.label.num_classes

    if result_mode == 'mean':
        average_scores = np.zeros(num_classes)
        for score in score_list:
            for idx in range(num_classes):
                average_scores[idx] += score[idx] / len(score_list)
    elif result_mode == 'min':
        score_list = sorted(score_list, key=lambda p: p[audio_label_idx], reverse=False)
        average_scores = score_list[0]
    elif result_mode == 'max':
        score_list = sorted(score_list, key=lambda p: p[audio_label_idx], reverse=True)
        average_scores = score_list[0]
    elif result_mode == 'average_duration_ms':
        average_window_length = 1 + average_window_duration_ms // timeshift_ms

        if len(score_list) > average_window_length:
            windows_number = 1 + len(score_list) - average_window_length

            # Calculate the average score across all the results in the window.
            average_scores_list = []
            for windows_idx in range(windows_number):
                score_list_window = score_list[windows_idx: windows_idx + average_window_length]
                average_scores = np.zeros(num_classes)
                for score in score_list_window:
                    for idx in range(num_classes):
                        average_scores[idx] += score[idx] / len(score_list_window)
                average_scores_list.append(average_scores)

            # Sort the averaged results.
            average_scores_list = sorted(average_scores_list, key=lambda p: p[audio_label_idx])
            average_scores = average_scores_list[0]
        else:
            average_scores = np.zeros(num_classes)
            for score in score_list:
                for idx in range(num_classes):
                    average_scores[idx] += score[idx] / len(score_list)

    else:
        raise Exception("[ERROR:] Unknow result mode, please check!")

    pred = np.argmax(average_scores)
    return pred, average_scores


def longterm_audio_predict(cfg, net, lmdb_env, audio_file, audio_mode, audio_label, background_data, add_noise_on, timeshift_ms, align_bool):
    """
    加载语音，前向传播预测结果
    """
    # init 
    input_dir = os.path.join(os.path.dirname(cfg.general.data_csv_path), 'dataset_audio', audio_mode)
    input_dir = os.path.join(input_dir, audio_label)
    sampling_rate = cfg.dataset.sampling_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sampling_rate * clip_duration_ms / 1000)

    # load data
    data = read_audio_lmdb(lmdb_env, audio_file)

    # alignment data
    data_length = len(data)
    data = np.pad(data, (max(0, (desired_samples - data_length)//2), 0), "constant")
    data = np.pad(data, (0, max(0, (desired_samples - data_length + 1)//2)), "constant")

    # align data：基于帧对齐模式，保证语音时间足够长，便于后续测试
    if align_bool:
        timeshift_samples = int(sampling_rate * timeshift_ms / 1000)
        desired_data_samples = desired_samples + 100 * timeshift_samples
        data = np.pad(data, (max(0, (desired_data_samples - data_length)//2), 0), "constant")
        data = np.pad(data, (0, max(0, (desired_data_samples - data_length + 1)//2)), "constant")

    # add noise 
    if audio_label == hparams.SILENCE_LABEL or add_noise_on:
        data = dataset_augmentation.dataset_add_noise(cfg, data, background_data, bool_force_add_noise=True)

    # gen data list
    # 基于滑窗的方式，获得数据列表
    data_list = []
    if len(data) > desired_samples:
        timeshift_samples = int(sampling_rate * timeshift_ms / 1000)
        data_number = 1 + (len(data) - desired_samples) // timeshift_samples
        for data_idx in range(data_number):
            data_list.append(data[timeshift_samples * data_idx: timeshift_samples * data_idx + desired_samples])
    else:
        data_list.append(data)
    
    # model predict
    score_list = []
    for data_idx in range(len(data_list)):
        score = model_predict(cfg, net, data_list[data_idx])
        score_list.append(score[0])

    return score_list


def infer(args, dataset_mode):
    """
    模型推理，通过滑窗的方式测试每一小段音频数据，随后进行后处理操作
    """
    # load configuration file
    cfg = load_cfg_file(args.input)

    # align bool
    align_bool = 'align' in args.input

    # init 
    num_classes = cfg.dataset.label.num_classes
    positive_label_together = cfg.dataset.label.positive_label_together
    negative_label_together = cfg.dataset.label.negative_label_together

    # define network
    net = import_network(cfg, cfg.net.model_name, cfg.net.class_name)

    # load prediction model
    load_checkpoint(net, 
                    cfg.general.load_mode_type,
                    cfg.general.save_dir, cfg.test.model_epoch, cfg.general.finetune_sub_folder_name,
                    cfg.general.finetune_model_path,
                    cfg.general.finetune_state_name, cfg.general.finetune_ignore_key_list, cfg.general.finetune_add_module_type)

    net.eval()

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

    # load data 
    data_pd = pd.read_csv(cfg.general.data_csv_path)
    data_pd_mode = data_pd[data_pd['mode'] == dataset_mode]
    data_file_list = data_pd_mode['file'].tolist()
    data_mode_list = data_pd_mode['mode'].tolist()
    data_label_list = data_pd_mode['label'].tolist()

    # lmdb
    lmdb_path = os.path.join(os.path.dirname(cfg.general.data_csv_path), 'dataset_audio_lmdb', '{}.lmdb'.format(dataset_mode))
    lmdb_env = load_lmdb_env(lmdb_path)

    # load background noise
    background_data = load_background_noise_lmdb(cfg)

    results_list = []
    preds = []
    labels = []
    for audio_idx in tqdm(range(len(data_file_list))):
        results_dict = {}
        results_dict['file'] = data_file_list[audio_idx]
        results_dict['mode'] = data_mode_list[audio_idx]
        results_dict['label'] = data_label_list[audio_idx]

        if positive_label_together and results_dict['label'] in cfg.dataset.label.positive_label:
            results_dict['label_idx'] = label_index[positive_label_together_label_list[0]]
        elif negative_label_together and results_dict['label'] in cfg.dataset.label.negative_label:
            results_dict['label_idx'] = label_index[negative_label_together_label_list[0]]
        else:
            results_dict['label_idx'] = label_index[results_dict['label']]
        assert results_dict['mode']  == dataset_mode, "[ERROR:] Something wronge about mode, please check"

        score_list = longterm_audio_predict(cfg, net, lmdb_env, results_dict['file'], results_dict['mode'], results_dict['label'], 
                                            background_data, args.add_noise_on, args.timeshift_ms, align_bool)
        
        if align_bool:
            pred, score = longterm_audio_align_post_processing(cfg, score_list, args.result_mode, args.timeshift_ms, args.average_window_duration_ms)
        else:
            pred, score = longterm_audio_post_processing(cfg, score_list, results_dict['label_idx'], args.result_mode, args.timeshift_ms, args.average_window_duration_ms)

        preds.append(pred)
        labels.append(results_dict['label_idx'])

        results_dict['result_idx'] = pred
        for classe_idx in range(num_classes):
            results_dict['prob_{}'.format(classe_idx)] = score[classe_idx]
        
        results_list.append(results_dict)

    # caltulate accuracy
    accuracy = float((np.array(preds) == np.array(labels)).astype(int).sum()) / float(len(labels))
    msg = '{}_accuracy: {:.4f}'.format(dataset_mode, accuracy)
    print(msg)

    # out csv
    csv_data_pd = pd.DataFrame(results_list)
    csv_data_pd.to_csv(os.path.join(cfg.general.save_dir, 'infer_longterm_{}_augmentation_{}_{}.csv'.format(dataset_mode, args.add_noise_on, args.result_mode)), index=False, encoding="utf_8_sig")


def main():
    """
    对音频文件进行测试，配置为 --input 中的 config 文件，当存在音频文件长度大于模型送入的音频文件长度时(1s\2s\3s), 该脚本会通过滑窗的方式测试每一小段音频数据，随后进行后处理操作。
    方式一：将每段结果的平均结果(或者对应 label 最小值)作为最终测试结果，该过程有悖于测试流程，存在误差。
    方式二：计算连续 800ms(27帧) 音频的平均值结果，在得到的平均结果中对应 label 最小值作为最终结果，该过程近似测试流程，可以作为参考。
    方式三：针对帧对齐方案，采取双边门限法进行后处理。
    """

    default_mode = "validation"     # ["testing,validation,training"]
    default_add_noise_on = False    # [True,False]
    # default_timeshift_ms = 30       # [30]
    # default_average_window_duration_ms = 800                   # [800, 1500] only for mode: average_duration_ms/double_edge_triggered_detecting
    default_timeshift_ms = 100       # [30, 100]
    default_average_window_duration_ms = 1000                    # [800, 1000, 1500] only for mode: average_duration_ms/double_edge_triggered_detecting
    default_result_mode = 'mean'     # ['min','mean','max', 'average_duration_ms'] align：["double_edge_triggered_detecting"]
    
    parser = argparse.ArgumentParser(description='Streamax KWS Infering Engine')
    # parser.add_argument('--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_tf_speech.py", help='config file')
    # parser.add_argument('--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", help='config file')
    # parser.add_argument('--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_align_xiaoyu.py", help='config file')
    # parser.add_argument('--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui16k.py", help='config file')
    # parser.add_argument('--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_activatebwc.py", help='config file')
    # parser.add_argument('--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_heybodycam.py", help='config file')
    parser.add_argument('--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoan8k.py", help='config file')
    # parser.add_argument('--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoan16k.py", help='config file')
    # parser.add_argument('--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_nihaoxiaoan8k.py", help='config file')
    # parser.add_argument('--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_nihaoxiaoan16k.py", help='config file')
    
    parser.add_argument('--mode', type=str, default=default_mode)
    parser.add_argument('--add_noise_on', type=bool, default=default_add_noise_on)
    parser.add_argument('--timeshift_ms', type=int, default=default_timeshift_ms)
    parser.add_argument('--average_window_duration_ms', type=int, default=default_average_window_duration_ms)
    parser.add_argument('--result_mode', type=str, default=default_result_mode)
    args = parser.parse_args()

    mode_list = args.mode.strip().split(',')
    for mode_type in mode_list:
        infer(args, mode_type)


if __name__ == "__main__":
    main()
