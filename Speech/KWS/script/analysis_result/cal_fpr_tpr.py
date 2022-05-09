import librosa
import os
import pandas as pd
import sys
from scipy.io import wavfile

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio

sys.path.insert(0, '/home/huanyuan/code/demo/common')
from common.utils.python.metrics_tools import *


def cal_fpr_tpr(src_csv, pst_csv, positive_label_list, bool_write_audio):
    # load csv
    src_pd = pd.read_csv(src_csv)
    try:
        pst_pd = pd.read_csv(pst_csv)
    except BaseException:
        print("Empty csv: {}".format(pst_csv))
        tn = len(src_pd[src_pd['label'] not in positive_label_list])
        fp = 0 
        fn = len(src_pd[src_pd['label'] in positive_label_list])
        tp = 0
        accuracy = get_accuracy(tn, fp, fn, tp)
        tpr = get_tpr(tn, fp, fn, tp)
        fpr = get_fpr(tn, fp, fn, tp)
        print("[Ground Truth] Accuracy:{:.2f}%({}/{}), Tpr:{:.2f}%({}/{}), Fpr:{:.2f}%({}/{})".format(accuracy*100, tp+tn, (tp+fp+tn+fn), tpr*100, tp, tp+fn, fpr*100, fp, fp+tn))
        print("[Confusion Matrix] \n[{}, {} \n {}, {}]".format(tp, fn, fp, tn))
        return tn, fp, fn, tp

    src_list = []
    for _, row in src_pd.iterrows():
        src_list.append({'label':row['label'], 'start_time':int(row['start_time']), 'end_time':int(row['end_time'])})
        assert row['start_time'] < row['end_time']

    pst_list = []   
    for _, row in pst_pd.iterrows():
        pst_list.append({'label':row['label'], 'start_time':int(row['start_time']), 'end_time':int(row['end_time']), 'matched':0})
        assert row['start_time'] < row['end_time']

    # init 
    y_true_list = []
    y_pred_list = []
    fn_list = []
    fp_list = []
    double_matched_list = []
    unmatched_list = []

    # match y_true_list/y_pred_list
    for idx in range(len(src_list)):
        row_idx = src_list[idx]

        # y_true_idx
        y_true_idx = 1 if row_idx['label'] in positive_label_list else 0
        y_true_list.append(y_true_idx)

        # y_pred_idx 
        y_pred_idx = 0
        for idy in range(len(pst_list)):
            row_idy = pst_list[idy]
            if (row_idx['start_time'] > row_idy['start_time'] and row_idx['start_time'] < row_idy['end_time'] and row_idy['label'] in positive_label_list) \
                or (row_idx['end_time'] > row_idy['start_time'] and row_idx['end_time'] < row_idy['end_time'] and row_idy['label'] in positive_label_list) \
                or (row_idx['start_time'] < row_idy['start_time'] and row_idx['end_time'] > row_idy['end_time'] and row_idy['label'] in positive_label_list):
                # 第一次匹配
                if y_pred_idx == 0 and row_idy['matched'] == 0:
                    row_idy['matched'] = 1
                    y_pred_idx = 1
                else:
                    # find double matched list，第二次匹配，说明两个检测结果与标签交叉
                    row_idy['matched'] = 1
                    y_pred_idx = 1
                    double_matched_list.append({'label':row_idy['label'], 'start_time':int(row_idy['start_time']), 'end_time':int(row_idy['end_time'])})
                
                # find fp list 
                if y_true_idx == 0 and y_pred_idx == 1:
                    fp_list.append({'label':row_idy['label'], 'start_time':int(row_idy['start_time']), 'end_time':int(row_idy['end_time'])})

        y_pred_list.append(y_pred_idx)

        # find fn list 
        if y_true_idx == 1 and y_pred_idx == 0:
            fn_list.append({'label':row_idx['label'], 'start_time':int(row_idx['start_time']), 'end_time':int(row_idx['end_time'])})

        # find fp list 
        if y_true_idx == 0 and y_pred_idx == 1:
            fp_list.append({'label':row_idx['label'], 'start_time':int(row_idx['start_time']), 'end_time':int(row_idx['end_time'])})

    # find unmatched detection result
    for idy in range(len(pst_list)):
        row_idy = pst_list[idy]
        if row_idy['matched'] != 1:
            # print("[Warning:] Please check result, no labels are found")
            unmatched_list.append({'label':row_idy['label'], 'start_time':int(row_idy['start_time']), 'end_time':int(row_idy['end_time'])})

    tn, fp, fn, tp = get_confusion_matrix(y_true_list, y_pred_list)
    accuracy = get_accuracy(tn, fp, fn, tp)
    tpr = get_tpr(tn, fp, fn, tp)
    fpr = get_fpr(tn, fp, fn, tp)
    print("[Ground Truth] Accuracy:{:.2f}%({}/{}), Tpr:{:.2f}%({}/{}), Fpr:{:.2f}%({}/{})".format(accuracy*100, tp+tn, (tp+fp+tn+fn), tpr*100, tp, tp+fn, fpr*100, fp, fp+tn))
    print("[Confusion Matrix] \n[{}, {} \n {}, {}]".format(tp, fn, fp, tn))
    print("[Detection Total] number:{}, matched number:{}, unmatched number:{}, double matched number:{}".format(
        len(pst_list), len(pst_list) - len(unmatched_list), len(unmatched_list), len(double_matched_list)))

    print()
    for fn_case in fn_list:
        print("[FN] {}".format(fn_case))
    print()
    for fp_case in fp_list:
        print("[FP] {}".format(fp_case)) 
    print()
    for double_matched_case in double_matched_list:
        print("[Double Matched] {}".format(double_matched_case))
    print()
    for unmatched_case in unmatched_list:
        print("[Unmatched] {}".format(unmatched_case))

    if bool_write_audio:
        # load data
        sampling_rate = 16000
        audio_data = librosa.core.load(src_csv.split('.')[0] + '.wav', sr=sampling_rate)[0]

        # mkdirs
        output_dir = os.path.join(os.path.dirname(pst_csv), 'audio_result')
        if not os.path.exists(output_dir):    
            os.makedirs(output_dir)

        for fn_case in fn_list:
            # mkdirs
            output_subdir = os.path.join(output_dir, 'fn')
            if not os.path.exists(output_subdir):    
                os.makedirs(output_subdir)

            output_path = os.path.join(output_subdir, 'label_{}_starttime_{}.wav'.format(fn_case['label'], fn_case['start_time']))
            start_time = int(sampling_rate * fn_case['start_time'] / 1000)
            end_time = int(sampling_rate * fn_case['end_time'] / 1000)
            output_wav = audio_data[start_time: end_time]
            audio.save_wav(output_wav, output_path, sampling_rate=sampling_rate)

        for fp_case in fp_list:
            # mkdirs
            output_subdir = os.path.join(output_dir, 'fp')
            if not os.path.exists(output_subdir):    
                os.makedirs(output_subdir)
                
            output_path = os.path.join(output_subdir, 'label_{}_starttime_{}.wav'.format(fp_case['label'], fp_case['start_time']))
            start_time = int(sampling_rate * fp_case['start_time'] / 1000)
            end_time = int(sampling_rate * fp_case['end_time'] / 1000)
            output_wav = audio_data[start_time: end_time]
            audio.save_wav(output_wav, output_path, sampling_rate=sampling_rate)

        for double_matched_case in double_matched_list:
            # mkdirs
            output_subdir = os.path.join(output_dir, 'double_matched')
            if not os.path.exists(output_subdir):    
                os.makedirs(output_subdir)
                
            output_path = os.path.join(output_subdir, 'label_{}_starttime_{}.wav'.format(double_matched_case['label'], double_matched_case['start_time']))
            start_time = int(sampling_rate * double_matched_case['start_time'] / 1000)
            end_time = int(sampling_rate * double_matched_case['end_time'] / 1000)
            output_wav = audio_data[start_time: end_time]
            audio.save_wav(output_wav, output_path, sampling_rate=sampling_rate)

        for unmatched_case in unmatched_list:
            # mkdirs
            output_subdir = os.path.join(output_dir, 'unmatched')
            if not os.path.exists(output_subdir):    
                os.makedirs(output_subdir)
                
            output_path = os.path.join(output_subdir, 'label_{}_starttime_{}.wav'.format(unmatched_case['label'], unmatched_case['start_time']))
            start_time = int(sampling_rate * unmatched_case['start_time'] / 1000)
            end_time = int(sampling_rate * unmatched_case['end_time'] / 1000)
            output_wav = audio_data[start_time: end_time]
            audio.save_wav(output_wav, output_path, sampling_rate=sampling_rate)

    return tn, fp, fn, tp


def cal_fpr_tpr_per_folder(src_folder, pst_folder, file_format_list, positive_label_list, bool_write_audio):
    file_list = os.listdir(src_folder)
    file_list.sort()

    total_tn, total_fp, total_fn, total_tp = 0, 0, 0, 0
    for idx in range(len(file_list)):
        file_name = file_list[idx]

        if not file_name.endswith('.wav'):
            continue
            
        # satisfy file_format
        bool_satisfy_file_format = True
        for file_format in file_format_list:
            if file_format not in file_name:
                bool_satisfy_file_format = False
        if not bool_satisfy_file_format:
            continue

        file_path = os.path.join(src_folder, file_name)
        src_csv = file_path.replace('.wav', '.csv')
        pst_csv = os.path.join(pst_folder, file_name.split('.')[0], 'found_words.csv')
        
        print("[Information] csv: {}".format(src_csv))
        tn, fp, fn, tp = cal_fpr_tpr(src_csv,
                                    pst_csv,
                                    positive_label_list,
                                    bool_write_audio)
        
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp
        print()

    total_accuracy = get_accuracy(total_tn, total_fp, total_fn, total_tp)
    total_tpr = get_tpr(total_tn, total_fp, total_fn, total_tp)
    total_fpr = get_fpr(total_tn, total_fp, total_fn, total_tp)
    print("[Ground Truth] Total Accuracy:{:.2f}%({}/{}), Tpr:{:.2f}%({}/{}), Fpr:{:.2f}%({}/{})".format(\
            total_accuracy*100, total_tp+total_tn, (total_tp+total_fp+total_tn+total_fn), total_tpr*100, total_tp, \
            total_tp+total_fn, total_fpr*100, total_fp, total_fp+total_tn))
    print("[Confusion Matrix] \n[{}, {} \n {}, {}]".format(total_tp, total_fn, total_fp, total_tn))
                                    
if __name__ == "__main__":
    # bool_write_audio = False
    bool_write_audio = True

    # xiaoyu
    # cal_fpr_tpr("/mnt/huanyuan/model/test_straming_wav/xiaoyu_12042020_testing_3600_001.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/{}/test_straming_wav/xiaoyu_12042020_testing_3600_001_threshold_{}/found_words.csv".format(model_name, "_".join(threshold.split('.'))),
    #             "xiaoyu",
    #             bool_write_audio)

    # # xiaorui
    # cal_fpr_tpr("/mnt/huanyuan/model/test_straming_wav/xiaorui_1_4_04302021_validation_3600.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws/kws_xiaorui/kws_xiaorui8k_56_196_1_0_resnet14_fbankcpu_06252021/test_straming_wav/xiaorui_1_4_04302021_validation_3600_threshold_0_5/found_words.csv",
    #             ["xiaorui", "xiaorui_16k"],
    #             bool_write_audio)

    # # xiaole
    # cal_fpr_tpr("/mnt/huanyuan/model/test_straming_wav/xiaole_11252020_testing_3600_001.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/{}/test_straming_wav/xiaole_11252020_testing_3600_001_threshold_0_95/found_words.csv".format(model_name),
    #             "xiaole",
    #             bool_write_audio)

    # xiaoan8k
    # cal_fpr_tpr("/mnt/huanyuan/model_final/test_straming_wav/xiaoan8k_1_3_04152021_validation.csv",
    #             "/mnt/huanyuan/model/kws/kws_xiaoan/kws_xiaoan8k_6_7_2s_tc_resnet14_fbankcpu_kd_04192022/test_straming_wav/xiaoan8k_1_3_04152021_validation_threshold_0_5/found_words.csv",
    #             ["xiaoanxiaoan_8k", "xiaoanxiaoan_16k"],
    #             bool_write_audio)
    cal_fpr_tpr_per_folder("/mnt/huanyuan2/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/货车怠速场景/更新漏标注数据处理/",
                            "/mnt/huanyuan/model/kws/kws_xiaoan/kws_xiaoan8k_6_7_2s_tc_resnet14_fbankcpu_kd_04192022/test_straming_wav/实车录制_0427_pytorch/阈值_05_05/货车怠速场景/",
                            # "/mnt/huanyuan/model/kws/kws_xiaoan/kws_xiaoan8k_6_7_2s_tc_resnet14_fbankcpu_kd_04192022/test_straming_wav/实车录制_0427_pytorch/阈值_09_05/货车怠速场景/",
                            ["_adpro"],
                            # ["_mic"],
                            # ["danbin_ori"],
                            # ["danbin_asr"],
                            ["xiaoanxiaoan_8k", "xiaoan8k", "Weakup"],
                            bool_write_audio)
    # cal_fpr_tpr("/mnt/huanyuan/data/speech/kws/xiaoan_dataset/test_dataset/实车录制_0427/货车怠速场景/更新漏标注数据处理/李煜_danbin_ori.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoan8k_1_8_res15_fbankcpu_041262021/test_straming_wav/实车录制_0427_pytorch/阈值_05_03/货车怠速场景/李煜_danbin_ori/found_words.csv",
    #             ["xiaoanxiaoan_8k", "xiaoan8k"],
    #             bool_write_audio)

    # # activatebwc
    # cal_fpr_tpr("/mnt/huanyuan/model/test_straming_wav/activatebwc_1_5_03312021_validation.csv",
    #             "/mnt/huanyuan/model_final/kws/kws_english/kws_activatebwc_2_8_tc-resnet14-amba_fbankcpu_kd_09292021/test_straming_wav/activatebwc_1_5_03312021_validation_threshold_0_8/found_words.csv",
    #             "activatebwc",
    #             bool_write_audio)
    # cal_fpr_tpr("/mnt/huanyuan/data/speech/kws/english_kws_dataset/test_dataset/海外同事录制_0425/安静场景/场景一/RM_KWS_ACTIVATEBWC_ovweseas_asr_S010M0D00T1.csv",
    #             "/mnt/huanyuan/data/speech/Recording_sample/demo_kws_asr_online_api/2021-04-25-14-53-09/RM_KWS_ACTIVATEBWC_ovweseas_asr_S010M0D00T1/found_words.csv",
    #             "activatebwc",
    #             bool_write_audio)
    # cal_fpr_tpr_per_folder("/mnt/huanyuan/data/speech/kws/english_kws_dataset/test_dataset/海外同事录制_0425/路边场景/场景一/",
    #                         "/mnt/huanyuan/model_final/kws/kws_english/kws_activatebwc_2_8_tc-resnet14-amba_fbankcpu_kd_09292021/test_straming_wav/海外同事录制_0425/阈值_05_05/路边场景/场景一/",
    #                         # ["RM_KWS_ACTIVATEBWC_ovweseas_ori_"],
    #                         ["RM_KWS_ACTIVATEBWC_ovweseas_asr_"],
    #                         "activatebwc",
    #                         bool_write_audio)

    # # positive
    # cal_fpr_tpr("/mnt/huanyuan/model/test_straming_wav/pretrain_1_1_12212020_validation_3600_001.csv",
    #             "/mnt/huanyuan/model/model_10_30_25_21/model/{}/test_straming_wav/pretrain_1_1_12212020_validation_3600_001_threshold_{}/found_words.csv".format(model_name, "_".join(threshold.split('.'))),
    #             "positive",
    #             bool_write_audio)