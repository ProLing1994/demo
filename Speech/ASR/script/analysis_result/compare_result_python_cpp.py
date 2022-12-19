import argparse
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/ASR')
from impl.asr_decode_pyimpl import get_edit_distance, get_edit_distance


def compare_result(args):
    python_pd = pd.read_csv(args.python_result)
    cpp_pd = pd.read_csv(args.cpp_result)
    merge_pd = pd.merge(python_pd, cpp_pd, on='data', how='inner')
    
    total_symbol = 0.0
    error_symbol = 0.0
    total_pinyin = 0.0
    error_pinyin = 0.0
    for idx, row in merge_pd.iterrows(): 
        total_symbol += max(len("".join(row['caffe_result'].strip().split(' '))), len("".join(row['amba_result'].strip().split(' '))))
        error_symbol += get_edit_distance("".join(row['caffe_result'].strip().split(' ')), "".join(row['amba_result'].strip().split(' ')))
        total_pinyin += max(len(row['caffe_result'].strip().split(' ')), len(row['amba_result'].strip().split(' ')))
        error_pinyin += get_edit_distance(row['caffe_result'], row['amba_result'])
        if error_pinyin != 0:
            print(row['caffe_result'].strip().split(' '))
            print(row['amba_result'].strip().split(' '))
            print("Symbol Edit Distance:", get_edit_distance("".join(row['caffe_result'].strip().split(' ')), "".join(row['amba_result'].strip().split(' '))))
            print("Pinyin Edit Distance:", get_edit_distance(row['caffe_result'], row['amba_result']))
    print("Error Symbol: {}/{}, {:.2f}%".format(error_symbol, total_symbol, error_symbol/total_symbol*100))
    print("Error Pinyin: {}/{}, {:.2f}%".format(error_pinyin, total_pinyin, error_pinyin/total_pinyin*100))


def main():
    # # chinese
    # default_python_result = "/home/huanyuan/share/audio_data/第三批数据/安静场景/result_caffe/result.csv"
    # default_cpp_result = "/home/huanyuan/share/audio_data/第三批数据/安静场景/result_amba/result.csv"

    # english
    # default_python_result = "/home/huanyuan/share/audio_data/english_wav/result_caffe/result_sliding_window_english_symbol.csv"
    # default_cpp_result = "/home/huanyuan/share/audio_data/english_wav/amba_test/result_english_symbol.csv"
    default_python_result = "/home/huanyuan/share/audio_data/english_wav/result_caffe/result_sliding_window_english_bpe.csv"
    default_cpp_result = "/home/huanyuan/share/audio_data/english_wav/amba_test/result_english_bpe.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument('--python_result', type=str, default=default_python_result)
    parser.add_argument('--cpp_result', type=str, default=default_cpp_result)
    args = parser.parse_args()
    compare_result(args)


if __name__ == '__main__':
    main()