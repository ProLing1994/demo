import argparse
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/ASR')
from impl.asr_decode_pympl import edit_distance_symbol, edit_distance_pinyin


def compare_result(args):
    python_pd = pd.read_csv(args.python_result)
    cpp_pd = pd.read_csv(args.cpp_result)
    merge_pd = pd.merge(python_pd, cpp_pd, on='data', how='inner')
    
    total_symbol = 0.0
    error_symbol = 0.0
    total_pinyin = 0.0
    error_pinyin = 0.0
    for idx, row in merge_pd.iterrows(): 
        total_symbol += max(len("".join(row['caffe_result'].split(' '))), len("".join(row['amba_result'].split(' '))))
        error_symbol += edit_distance_symbol("".join(row['caffe_result'].split(' ')), "".join(row['amba_result'].split(' ')))
        total_pinyin += max(len(row['caffe_result'].split(' ')), len(row['amba_result'].split(' ')))
        error_pinyin += edit_distance_pinyin(row['caffe_result'], row['amba_result'])
        if row['caffe_result'] != row['amba_result']:
            print(row['caffe_result'])
            print(row['amba_result'])
            print("Symbol Edit Distance:", edit_distance_symbol("".join(row['caffe_result'].split(' ')), "".join(row['amba_result'].split(' '))))
            print("Pinyin Edit Distance:", edit_distance_pinyin(row['caffe_result'], row['amba_result']))
    print("Error Symbol: {}/{}, {:.2f}%".format(error_symbol, total_symbol, error_symbol/total_symbol*100))
    print("Error Pinyin: {}/{}, {:.2f}%".format(error_pinyin, total_pinyin, error_pinyin/total_pinyin*100))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--python_result', type=str, default="/home/huanyuan/share/audio_data/第三批数据/安静场景/result_caffe/result.csv")
    parser.add_argument('--cpp_result', type=str, default="/home/huanyuan/share/audio_data/第三批数据/安静场景/result_amba/result.csv")
    args = parser.parse_args()
    compare_result(args)

if __name__ == '__main__':
    main()