import argparse
import os 
import pandas as pd

def find_no_use_wav(input_dir, input_csv_list):
    file_list = os.listdir(input_dir)

    # load used file
    used_file_list = []
    for input_csv in input_csv_list:
        used_file_pd = pd.read_csv(input_csv)
        for _, row in used_file_pd.iterrows():
            used_file_list.append(os.path.basename(row['file'])) 

    # load no used file 
    no_used_file_list = []
    for file_name in file_list:
        file_path = os.path.join(input_dir, file_name)
        if file_name not in used_file_list:
            no_used_file_dict = {'file': file_path}
            no_used_file_list.append(no_used_file_dict)
    
    # mkdir 
    output_dir = os.path.join(os.path.dirname(os.path.dirname(input_csv_list[0])), 'noused_in_test_straming_wav')
    if not os.path.exists(output_dir):    
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, os.path.basename(input_csv_list[0]).split('.')[0][:-4] + '_noused.csv')
    output_csv_pd = pd.DataFrame(no_used_file_list)
    output_csv_pd.to_csv(output_path, index=False)


def main():
    # default_input_dir = '/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/baijiajiangtan'
    # default_input_csv_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_baijiajiangtan_21600_001.csv']
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/jinpingmei/"
    # default_input_csv_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_jinpingmei_7200_001.csv']
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/yeshimiwen/"
    # default_input_csv_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_yeshimiwen_43200_001.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_yeshimiwen_43200_002.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_yeshimiwen_43200_003.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_yeshimiwen_43200_004.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_yeshimiwen_43200_005.csv']
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/history/zhongdongwangshi/"
    # default_input_csv_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_history_zhongdongwangshi_7200_001.csv']
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/news/cishicike/"
    # default_input_csv_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_news_cishicike_43200_001.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_news_cishicike_43200_002.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_news_cishicike_43200_003.csv']
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/music_station/qingtingkongzhongyinyuebang/"
    # default_input_csv_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_001.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_002.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_003.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_004.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_005.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_006.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_007.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_008.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_009.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_station_qingtingkongzhongyinyuebang_43200_010.csv',]
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/novel/douluodalu/"
    # default_input_csv_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_001.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_002.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_003.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_004.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_005.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_006.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_007.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_008.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_009.csv',
    #                             '/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_novel_douluodalu_43200_010.csv',]
    # default_input_dir =  "/mnt/huanyuan/data/speech/Negative_sample/QingTingFM/music/xingetuijian/"
    # default_input_csv_list = ['/mnt/huanyuan/data/speech/Negative_sample/test_straming_wav/QingTingFM_music_xingetuijian_21600_001.csv']

    parser = argparse.ArgumentParser(description="Prepare XiaoYu Dataset")
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--input_csv_list', type=str, default=default_input_csv_list)
    args = parser.parse_args()
    
    find_no_use_wav(args.input_dir, args.input_csv_list)

if __name__ == "__main__":
    main()