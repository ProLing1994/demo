import librosa
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/common')
from common.utils.python.metrics_tools import *


def cal_fpr_tpr(src_csv, pst_csv, positive_label, bool_write_audio):
    # laod csv
    src_pd = pd.read_csv(src_csv)
    try:
        pst_pd = pd.read_csv(pst_csv)
    except BaseException:
        print("Empty csv: {}".format(pst_csv))
        return 

    src_list = []
    for _, row in src_pd.iterrows():
        src_list.append({'label':row['label'], 'start_time':int(row['start_time']), 'end_time':int(row['end_time'])})
        # src_list.append({'label':row['lable'], 'start_time':int(row['start_time']), 'end_time':int(row['end_time'])})
        assert row['start_time'] < row['end_time']

    pst_list = []   
    for _, row in pst_pd.iterrows():
        pst_list.append({'label':row['label'], 'start_time':int(row['start_time']), 'end_time':int(row['end_time']), 'matched':0})
        assert row['start_time'] < row['end_time']

    # init 
    y_true = []
    y_pred = []
    fn_list = []
    fp_list = []
    double_matched_list = []
    sample_rate = 16000

    # match y_true/y_pred
    for idx in range(len(src_list)):
        row_idx = src_list[idx]

        # y_true_idx
        y_true_idx = 1 if row_idx['label'] == positive_label else 0
        y_true.append(y_true_idx)

        # y_pred_idx 
        y_pred_idx = 0
        for idy in range(len(pst_list)):
            row_idy = pst_list[idy]
            if (row_idx['start_time'] > row_idy['start_time'] and row_idx['start_time'] < row_idy['end_time'] and row_idy['label'] == positive_label) \
                or (row_idx['end_time'] > row_idy['start_time'] and row_idx['end_time'] < row_idy['end_time'] and row_idy['label'] == positive_label) \
                or (row_idx['start_time'] < row_idy['start_time'] and row_idx['end_time'] > row_idy['end_time'] and row_idy['label'] == positive_label):
                if y_pred_idx == 0 and row_idy['matched'] == 0:
                    row_idy['matched'] = 1
                    y_pred_idx = 1
                else:
                    # 找到两次结果，说明两个检测结果与标签交叉
                    row_idy['matched'] = 1
                    y_pred_idx = 1
                    double_matched_list.append({'label':row_idy['label'], 'start_time':int(row_idy['start_time']), 'end_time':int(row_idy['end_time'])})
                    print("[Warning:] Please check result, two results are found, indicating that the two test results cross the label")
        y_pred.append(y_pred_idx)

        # find fn list 
        if y_true_idx == 1 and y_pred_idx == 0:
            fn_list.append({'label':row_idx['label'], 'start_time':int(row_idx['start_time']), 'end_time':int(row_idx['end_time'])})

        # find fp list 
        if y_true_idx == 0 and y_pred_idx == 1:
            fp_list.append({'label':row_idx['label'], 'start_time':int(row_idx['start_time']), 'end_time':int(row_idx['end_time'])})

    assert len(y_true) == len(y_pred)
    tn, fp, fn, tp = get_confusion_matrix(y_true, y_pred)
    accuracy = get_accuracy(tn, fp, fn, tp)
    tpr = get_tpr(tn, fp, fn, tp)
    fpr = get_fpr(tn, fp, fn, tp)
    print("[Ground Truth] Accuracy:{:.2f}%({}/{}), Tpr:{:.2f}%({}/{}), Fpr:{:.2f}%({}/{})".format(accuracy*100, tp+tn, (tp+fp+tn+fn), tpr*100, tp, tp+fn, fpr*100, fp, fp+tn))
    print("[Confusion Matrix] \n[{}, {} \n {}, {}]".format(tp, fn, fp, tn))
    # print("[Ground Truth Total] number:{}, tp:{}, fp:{}, tn:{}, fn:{}".format((tp+fp+tn+fn), tp, fp, tn, fn))
    
    if bool_write_audio:
        # load data
        audio_data = librosa.core.load(src_csv.split('.')[0] + '.wav', sr=sample_rate)[0]

        output_dir = os.path.join(os.path.dirname(pst_csv), 'audio_result')
        if not os.path.exists(output_dir):    
            os.makedirs(output_dir)

    print()
    for fn_case in fn_list:
        print("[FN] {}".format(fn_case))

        if bool_write_audio:
            output_subdir = os.path.join(output_dir, 'fn')
            if not os.path.exists(output_subdir):    
                os.makedirs(output_subdir)
            output_path = os.path.join(output_subdir, 'label_{}_starttime_{}.wav'.format(fn_case['label'], fn_case['start_time']))
            start_time = int(sample_rate * fn_case['start_time'] / 1000)
            end_time = int(sample_rate * fn_case['end_time'] / 1000)
            output_wav = audio_data[start_time: end_time]
            librosa.output.write_wav(output_path, output_wav, sr=sample_rate)

    print()
    for fp_case in fp_list:
        print("[FP] {}".format(fp_case))

        if bool_write_audio:
            output_subdir = os.path.join(output_dir, 'fp')
            if not os.path.exists(output_subdir):    
                os.makedirs(output_subdir)
            output_path = os.path.join(output_subdir, 'label_{}_starttime_{}.wav'.format(fp_case['label'], fp_case['start_time']))
            start_time = int(sample_rate * fp_case['start_time'] / 1000)
            end_time = int(sample_rate * fp_case['end_time'] / 1000)
            output_wav = audio_data[start_time: end_time]
            librosa.output.write_wav(output_path, output_wav, sr=sample_rate)

    # find unmatched detection result
    unmatched_list = []
    for idy in range(len(pst_list)):
        row_idy = pst_list[idy]
        if row_idy['matched'] != 1:
            print("[Warning:] Please check result, no labels are found")
            unmatched_list.append({'label':row_idy['label'], 'start_time':int(row_idy['start_time']), 'end_time':int(row_idy['end_time'])})
    
    print()
    print("[Detection Total] number:{}, matched number:{}, unmatched number:{}, double matched number:{}".format(
        len(pst_list), len(pst_list) - len(unmatched_list), len(unmatched_list), len(double_matched_list)))

    for double_matched_case in double_matched_list:
        print("[Double Matched] {}".format(double_matched_case))

    for unmatched_case in unmatched_list:
        print("[Unmatched] {}".format(unmatched_case))
   

if __name__ == "__main__":
    bool_write_audio = True
    model_name = "kws_xiaole1_0_res15_11242020"

    # xiaoyu
    # cal_fpr_tpr("/mnt/huanyuan/model/test_straming_wav/xiaoyu_10292020_testing_3600_001.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/{}/test_straming_wav/xiaoyu_10292020_testing_3600_001_threshold_0_95/found_words.csv".format(model_name),
    #             "xiaoyu",
    #             bool_write_audio)

    # xiaorui
    # cal_fpr_tpr("/mnt/huanyuan/model/test_straming_wav/xiaorui_12032020_validation_3600_001.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/{}/test_straming_wav/xiaorui_12032020_validation_3600_001_threshold_0_8/found_words.csv".format(model_name),
    #             "xiaorui",
    #             bool_write_audio)

    # xiaole
    cal_fpr_tpr("/mnt/huanyuan/model/test_straming_wav/xiaole_11252020_testing_3600_001.csv",
                "/mnt/huanyuan/model/model_10_30_25_21/model/{}/test_straming_wav/xiaole_11252020_testing_3600_001_threshold_0_95/found_words.csv".format(model_name),
                "xiaole",
                bool_write_audio)