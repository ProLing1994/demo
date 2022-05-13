import argparse
import os
import pandas as pd

def time_merge(args):
    # pd init
    res_data_list = []
    res_data_dict = {}
    res_data_dict['time'] = 0
    
    # data_pd
    data_pd = pd.read_csv(args.csv_path, encoding='utf_8_sig')

    for idx, row in data_pd.iterrows():
        time_idx = row['time']
        start_time_idx = int(time_idx.split('-')[1])
        end_time_idx = int(time_idx.split('-')[2])
        
        if len(res_data_list):
            res_time = res_data_list[-1]['time']
            res_date_time = int(res_time.split('-')[0])
            res_start_time = int(res_time.split('-')[1])
            res_end_time = int(res_time.split('-')[2])

            if res_end_time + 2 * args.time_shift_s >= start_time_idx:
                res_start_time = min( res_start_time, start_time_idx )
                res_end_time = max( res_end_time, end_time_idx )
                res_data_list[-1]['time'] = '-'.join([str(res_date_time), str(res_start_time), str(res_end_time)])
            else:
                res_data_dict = {}
                res_data_dict['time'] = row['time']
                res_data_list.append(res_data_dict)
        else:
            res_data_dict = {}
            res_data_dict['time'] = row['time']
            res_data_list.append(res_data_dict)
    
    # out csv
    csv_data_pd = pd.DataFrame(res_data_list)
    csv_data_pd.to_csv(args.out_csv_path, index=False, encoding="utf_8_sig")


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.csv_path = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_TQ/264原始视频/5M_16mm_6M_白_0506_1000_1010.csv"
    args.out_csv_path = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_TQ/264原始视频/5M_16mm_6M_白_0506_1000_1010_time_merge.csv"

    # 截取视频段，前后扩展时间
    args.time_shift_s = 3

    time_merge(args)

if __name__ == '__main__':
    main()