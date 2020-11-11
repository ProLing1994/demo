import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd


def plot_line2d(plot_x, plot_y_src, plot_y_pst):
    plt.plot(plot_x, plot_y_src, color = 'r',  linewidth=1.0, linestyle='--', marker = 'o', label = 'label')
    plt.plot(plot_x, plot_y_pst, color = 'b',  linewidth=1.0, linestyle='--', marker = 'o', label = 'pred')
    plt.legend(loc=4)
    # plt.xlim([-0.01, 0.1])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.title('Score Line')


def plot_line3d(plot_x, plot_y_src_positive, plot_y_src_negative, plot_y_pst):
    plt.plot(plot_x, plot_y_src_positive, color = 'r',  linewidth=1.0, linestyle='--', marker = 'o', label = 'positive_label')
    plt.plot(plot_x, plot_y_src_negative, color = 'g',  linewidth=1.0, linestyle='--', marker = 'o', label = 'nagetive_label')
    plt.plot(plot_x, plot_y_pst, color = 'b',  linewidth=1.0, linestyle='--', marker = 'o', label = 'pred')
    plt.legend(loc=4)
    # plt.xlim([-0.01, 0.1])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.title('Score Line')


def show_score_line(src_csv, pst_csv, positive_label):
    output_dir = pst_csv.split('.')[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    src_pd = pd.read_csv(src_csv)
    pst_pd = pd.read_csv(pst_csv)

    src_list = []
    for _, row in src_pd.iterrows():
        src_list.append({'label':row['label'], 'start_time':int(row['start_time']), 'end_time':int(row['end_time'])})

    pst_dict = {}
    for _, row in pst_pd.iterrows():
        pst_dict[row['start_time']] = row['score']

    start_time = 0
    end_time = 0
    for idx in range(len(src_list)):
        # decide end time
        if idx != len(src_list) - 1:
            end_time = src_list[idx+1]['start_time']
        else:
            end_time = src_list[idx]['end_time']

        start_time = (start_time // 30) * 30
        end_time = (end_time // 30) * 30
        plot_x = np.arange(start_time, end_time, 30)
        plot_y_src_positive = np.zeros(plot_x.shape)
        plot_y_src_negative = np.zeros(plot_x.shape)
        plot_y_pst = np.zeros(plot_x.shape)
        for x_idx in range(plot_x.shape[0]):
            if plot_x[x_idx] > src_list[idx]['start_time'] and plot_x[x_idx] < src_list[idx]['end_time'] and src_list[idx]['label'] == positive_label:
                plot_y_src_positive[x_idx] = 1

            if plot_x[x_idx] > src_list[idx]['start_time'] and plot_x[x_idx] < src_list[idx]['end_time'] and src_list[idx]['label'] != positive_label:
                plot_y_src_negative[x_idx] = 1
            
            if plot_x[x_idx] in pst_dict:
                plot_y_pst[x_idx] = pst_dict[plot_x[x_idx]]

        plt.figure()
        # plot_line2d(plot_x, plot_y_src_positive, plot_y_pst)
        plot_line3d(plot_x, plot_y_src_positive, plot_y_src_negative, plot_y_pst)
        # plt.show()
        plt.savefig(os.path.join(output_dir, '{}_{}_{}_{}'.format(idx, start_time, end_time, src_list[idx]['label'])), dpi=300)
        plt.close()

        start_time = src_list[idx]['end_time']
    return

def show_score_line_find_words(src_csv, pst_csv, find_words_csv, output_dir, positive_label):
    output_dir = os.path.join(output_dir, os.path.basename(pst_csv).split('.')[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    src_pd = pd.read_csv(src_csv)
    pst_pd = pd.read_csv(pst_csv)
    find_words_pd = pd.read_csv(find_words_csv)

    src_list = []
    for _, row in src_pd.iterrows():
        src_list.append({'label':row['label'], 'start_time':int(row['start_time']), 'end_time':int(row['end_time'])})

    pst_dict = {}
    for _, row in pst_pd.iterrows():
        pst_dict[row['start_time']] = row['score']

    find_words_list = []
    for _, row in find_words_pd.iterrows():
        find_words_list.append({'label':row['label'], 'start_time':int(row['start_time']), 'end_time':int(row['end_time'])})
        

    start_time = 0
    end_time = 0
    for idx in range(len(find_words_list)):

        start_time = find_words_list[idx]['start_time']
        end_time = find_words_list[idx]['end_time']

        start_time = (start_time // 30) * 30
        end_time = (end_time // 30) * 30
        plot_x = np.arange(start_time, end_time, 30)
        plot_y_src_positive = np.zeros(plot_x.shape)
        plot_y_src_negative = np.zeros(plot_x.shape)
        plot_y_pst = np.zeros(plot_x.shape)
        for x_idx in range(plot_x.shape[0]):
            for src_idx in range(len(src_list)):
                if plot_x[x_idx] > src_list[src_idx]['start_time'] and plot_x[x_idx] < src_list[src_idx]['end_time'] and src_list[src_idx]['label'] == positive_label:
                    plot_y_src_positive[x_idx] = 1
                    break

                if plot_x[x_idx] > src_list[src_idx]['start_time'] and plot_x[x_idx] < src_list[src_idx]['end_time'] and src_list[src_idx]['label'] != positive_label:
                    plot_y_src_negative[x_idx] = 1
                    break

            if plot_x[x_idx] in pst_dict:
                plot_y_pst[x_idx] = pst_dict[plot_x[x_idx]]

        plt.figure()
        # plot_line2d(plot_x, plot_y_src_positive, plot_y_pst)
        plot_line3d(plot_x, plot_y_src_positive, plot_y_src_negative, plot_y_pst)
        # plt.show()
        plt.savefig(os.path.join(output_dir, '{}_label_{}_starttime_{}.jpg'.format(os.path.basename(os.path.dirname(pst_csv)), 
                                                                                find_words_list[idx]['label'], find_words_list[idx]['start_time'])), 
                    dpi=300)
        plt.close()
    return

if __name__ == "__main__":
    # show_score_line("/mnt/huanyuan/model/test_straming_wav/xiaoyu_10292020_testing_3600_001.csv", 
    #                 "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu4_1_fbank_timeshift_up_spec_on_focal_res15_11032020/test_straming_wav/xiaoyu_10292020_testing_3600_001_threshold_0_95/mean_scores.csv",
    #                 "xiaoyu")
    show_score_line("/mnt/huanyuan/model/test_straming_wav/weiboyulu_test_3600_001.csv", 
                    "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu4_1_fbank_timeshift_up_spec_on_focal_res15_11032020/test_straming_wav/weiboyulu_test_3600_001_threshold_0_95/mean_scores.csv",
                    "xiaoyu")
    # show_score_line_find_words("/mnt/huanyuan/model/test_straming_wav/xiaoyu_10292020_testing_3600_001.csv",
    #                             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu4_1_fbank_timeshift_up_spec_on_focal_res15_11032020/test_straming_wav/xiaoyu_10292020_testing_3600_001_threshold_0_95/mean_scores.csv",
    #                             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu4_1_fbank_timeshift_up_spec_on_focal_res15_11032020/test_straming_wav/xiaoyu_10292020_testing_3600_001_threshold_0_95/found_words.csv",
    #                             "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu4_1_fbank_timeshift_up_spec_on_focal_res15_11032020/test_straming_wav/xiaoyu_10292020_testing_3600_001_threshold_0_95/error_results/",
    #                             "xiaoyu")