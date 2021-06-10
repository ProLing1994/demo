import argparse
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/ASR')
from impl.asr_decode_pyimpl import get_edit_distance, get_edit_distance

def clean_tone(srt):
    srt = srt.replace('1', '')
    srt = srt.replace('2', '')
    srt = srt.replace('3', '')
    srt = srt.replace('4', '')
    srt = srt.replace('5', '')
    return srt

def analysis_edit_distance(args):
    python_pd = pd.read_csv(args.python_result)

    ground_truth_dict = {}
    with open(args.ground_truth, "r", encoding='utf-8') as f :
        lines = f.readlines()
        for line in lines:
            data_name = line.strip().split('\t')[0]
            ground_truth = line.strip().split('\t')[1]
            ground_truth = clean_tone(ground_truth)
            ground_truth_dict[data_name] = ground_truth
    
    total_symbol = 0.0
    error_symbol = 0.0
    total_pinyin = 0.0
    error_pinyin = 0.0
    for idx, row in python_pd.iterrows(): 
        result_idx = row['caffe_result']
        ground_truth_idx = ground_truth_dict[row['data']]
        
        total_symbol += len("".join(ground_truth_idx.split(' ')))
        error_symbol += get_edit_distance("".join(result_idx.split(' ')), "".join(ground_truth_idx.split(' ')))
        total_pinyin += len(ground_truth_idx.split(' '))
        error_pinyin += get_edit_distance(result_idx, ground_truth_idx)
        if result_idx != ground_truth_idx:
            print("Ground Truth:", ground_truth_idx)
            print("Result      :", result_idx)
            print("Symbol Edit Distance:", get_edit_distance("".join(result_idx.split(' ')), "".join(ground_truth_idx.split(' '))))
            print("Pinyin Edit Distance:", get_edit_distance(result_idx, ground_truth_idx))
    print("Error Symbol: {}/{}, {:.2f}%".format(error_symbol, total_symbol, error_symbol/total_symbol*100))
    print("Error Pinyin: {}/{}, {:.2f}%".format(error_pinyin, total_pinyin, error_pinyin/total_pinyin*100))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--python_result', type=str, default="/home/huanyuan/share/audio_data/第三批数据/安静场景/result_caffe/result_total_window.csv")
    parser.add_argument('--ground_truth', type=str, default="/home/huanyuan/share/audio_data/第三批数据/安静场景/transcript.txt")
    args = parser.parse_args()
    analysis_edit_distance(args)

if __name__ == '__main__':
    main()