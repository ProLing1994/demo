import argparse
import pandas as pd
import pickle
import sys
import torch.nn.functional as F

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from impl.pred_pyimpl import kws_load_model, load_background_noise, dataset_add_noise, model_predict
from dataset.kws.dataset_helper import *
from utils.train_tools import *


def longterm_audio_predict(cfg, net, audio_file, background_data, add_noise_on, timeshift_ms, result_mode):
    # init 
    num_classes = cfg.dataset.label.num_classes
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)

    # load data
    # data = librosa.core.load("/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu3_3_timeshift_spec_on_focal_res15_11032020/test_straming_wav/weiboyulu_test_3600_001_threshold_0_95/label_xiaoyu_starttime_1823970.wav", sr=sample_rate)[0]
    f = open(os.path.join(audio_file), 'rb')
    data = pickle.load(f)
    f.close()

    # alignment data
    data = np.pad(data, (0, max(0, desired_samples - len(data))), "constant")

    # add noise 
    if add_noise_on:
        data = dataset_add_noise(cfg, data, background_data, bool_silence_label=True)

    # calculate the average score across all the results
    data_list = []
    if len(data) > desired_samples:
        timeshift_samples = int(sample_rate * timeshift_ms / 1000)
        data_number = 1 + (len(data) - desired_samples) // timeshift_samples
        for data_idx in range(data_number):
            data_list.append(data[timeshift_samples * data_idx: timeshift_samples * data_idx + desired_samples])
    else:
        data_list.append(data)
    
    score_list = []
    for data_idx in range(len(data_list)):
        # # debug
        # librosa.output.write_wav(os.path.join("/home/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu_res15_10272020/testing/", str(data_idx) + '.wav'),  
        #     data_list[data_idx], sr=sample_rate)
        score = model_predict(cfg, net, data_list[data_idx])
        score_list.append(score[0])

    if result_mode == 'mean':
        average_scores = np.zeros(num_classes)
        for score in score_list:
            for idx in range(num_classes):
                average_scores[idx] += score[idx] / len(score_list)
    elif result_mode == 'min':
        # init 
        score_list = sorted(score_list, key=lambda p: p[1])
        average_scores = score_list[0]
    else:
        raise Exception("[ERROR:] Unknow result mode, please check!")

    pred = np.argmax(average_scores)
    return pred, average_scores


def predict(config_file, epoch, input_folder, add_noise_on, timeshift_ms, result_mode):
    # load configuration file
    cfg = load_cfg_file(config_file)

    # init 
    num_classes = cfg.dataset.label.num_classes

    # load prediction model
    model = kws_load_model(cfg.general.save_dir, int(cfg.general.gpu_ids), epoch)
    net = model['prediction']['net']
    net.eval()

    # load label index 
    label_index = load_label_index(cfg.dataset.label.positive_label, cfg.dataset.label.negative_label)

    # load data 
    data_file_list = os.listdir(input_folder)

    # load background noise
    background_data = load_background_noise(cfg)

    results_list = []
    preds = []
    labels = []
    for audio_idx in tqdm(range(len(data_file_list))):
        results_dict = {}
        results_dict['file'] = os.path.join(input_folder, data_file_list[audio_idx])
        results_dict['label'] = UNKNOWN_WORD_LABEL
        results_dict['label_idx'] = label_index[UNKNOWN_WORD_LABEL]
        
        pred, score = longterm_audio_predict(cfg, net, results_dict['file'], background_data, add_noise_on, timeshift_ms, result_mode)
        
        preds.append(pred)
        labels.append(results_dict['label_idx'])

        results_dict['result_idx'] = pred
        for classe_idx in range(num_classes):
            results_dict['prob_{}'.format(classe_idx)] = score[classe_idx]
        
        results_list.append(results_dict)

    # caltulate accuracy
    accuracy = float((np.array(preds) == np.array(labels)).astype(int).sum()) / float(len(labels))
    msg = 'epoch: {}, batch: {}, accuracy: {:.4f}'.format(model['prediction']['epoch'], model['prediction']['batch'], accuracy)
    print(msg)

    # out csv
    csv_data_pd = pd.DataFrame(results_list)
    csv_data_pd.to_csv(os.path.join(cfg.general.save_dir, 'infer_difficult_sample_mining_11112020.csv'), index=False, encoding="utf_8_sig")


def main():
    """
    使用模型对音频文件进行测试，配置为 --input 中的 config 文件，当存在音频文件长度大于模型送入的音频文件长度时(1s\2s\3s), 该脚本会通过滑窗的方式测试每一小段音频数据，将每段结果的平均结果(或者对应label最小值)作为最终测试结果，
    该过程有悖于测试流程，存在误差
    """
    default_input_folder = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_11032020/difficult_sample_mining_11112020/preload_data/"
    default_model_epoch = -1
    # default_add_noise_on = True
    default_add_noise_on = False
    default_timeshift_ms = 30
    # default_result_mode = 'mean'
    default_result_mode = 'min'

    parser = argparse.ArgumentParser(description='Streamax KWS Infering Engine')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config.py", help='config file')
    # parser.add_argument('--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", help='config file')
    parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu_2.py", help='config file')
    parser.add_argument('--input_folder', type=str, default=default_input_folder)
    parser.add_argument('--epoch', type=str, default=default_model_epoch)
    parser.add_argument('--add_noise_on', type=bool, default=default_add_noise_on)
    parser.add_argument('--timeshift_ms', type=int, default=default_timeshift_ms)
    parser.add_argument('--result_mode', type=str, default=default_result_mode)
    args = parser.parse_args()

    predict(args.config_file, args.epoch, args.input_folder, args.add_noise_on, args.timeshift_ms, args.result_mode)


if __name__ == "__main__":
    main()
