import argparse
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import sys

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from TTS.script.dataset_align.kaldi_tools import get_words_set


def analysis_prosody_BZNSYP(args):

    words_set = get_words_set(args.ctm_file)

    # pd
    data_pd = pd.read_csv(args.dataset_csv)
    file_list = data_pd['file'].tolist()

    # list，统计韵律节奏时长
    # 粗略统计，#1\#2\#3\#4 时长
    # 统计方法，出现韵律符号，前一个字所占用的时长
    time = [[], [], [], []]

    for _, row in tqdm(data_pd.iterrows(), total=len(file_list)):
        utterance = row["utterance"]
        text = row["text"]

        if not utterance in words_set:
            continue

        words = words_set[utterance]

        # text，去掉无用字符
        text = re.sub('[“”、，。：；？！—…#（）]', '', text)
        text = text.strip()
        text_no_num = re.sub('[1234]', '', text)

        words_idx = 0
        for text_idx in range(len(text)):
            if text[text_idx] in ['1', '2', '3', '4']:
                assert text_idx - 1 >= 0
                assert words[words_idx - 1][0] == text_no_num[words_idx - 1] or words[words_idx - 1][0] == '<UNK>'
                time[int(text[text_idx]) - 1].append(words[words_idx - 1][2])
                time[int(text[text_idx]) - 1].sort()
            elif words[words_idx][0] == text_no_num[words_idx]:
                words_idx += 1
            else:
                if words[words_idx][0] == '<UNK>':
                    if words[words_idx + 1][0] == text_no_num[words_idx + 1]:
                        words_idx += 1
                    elif words[words_idx + 1][0] == text_no_num[words_idx]:
                        if words_idx - 1 >= 0:
                            words[words_idx - 1][2] = str(float(words[words_idx - 1][2]) + float(words[words_idx][2]))
                        del words[words_idx]
                    else:
                        print()
                else:
                    print()
                
    # # type_1
    # plt.subplot(221)
    # time_np = np.array(time[0]).astype(np.float64)
    # plt.hist(x = time_np, bins = 10, color = 'steelblue', edgecolor = 'black')
    # plt.xlabel('#1: min: {}, max: {}, mean: {:.2f}'.format(min(time_np), max(time_np), np.mean(time_np)))

    # plt.subplot(222)
    # time_np = np.array(time[1]).astype(np.float64)
    # plt.hist(x = time_np, bins = 10, color = 'steelblue', edgecolor = 'black')
    # plt.xlabel('#2: min: {}, max: {}, mean: {:.2f}'.format(min(time_np), max(time_np), np.mean(time_np)))

    # plt.subplot(223)
    # time_np = np.array(time[2]).astype(np.float64)
    # plt.hist(x = time_np, bins = 10, color = 'steelblue', edgecolor = 'black')
    # plt.xlabel('#3: min: {}, max: {}, mean: {:.2f}'.format(min(time_np), max(time_np), np.mean(time_np)))

    # plt.subplot(224)
    # time_np = np.array(time[3]).astype(np.float64)
    # plt.hist(x = time_np, bins = 10, color = 'steelblue', edgecolor = 'black')
    # plt.xlabel('#4: min: {}, max: {}, mean: {:.2f}'.format(min(time_np), max(time_np), np.mean(time_np)))

    # type_2
    time_np = np.array(time[0]).astype(np.float64)
    plt.hist(x = time_np, bins = 10, color = 'red', alpha=0.5, edgecolor = 'black')

    time_np = np.array(time[1]).astype(np.float64)
    plt.hist(x = time_np, bins = 10, color = 'green', alpha=0.5, edgecolor = 'black')

    time_np = np.array(time[2]).astype(np.float64)
    plt.hist(x = time_np, bins = 10, color = 'blue', alpha=0.5, edgecolor = 'black')

    time_np = np.array(time[3]).astype(np.float64)
    plt.hist(x = time_np, bins = 10, color = 'orange', alpha=0.5, edgecolor = 'black')
    plt.xticks(np.arange(0, 0.7, 0.01), rotation = 45)

    # 显示图形
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', type=str,  default="/mnt/huanyuan2/data/speech/tts/Chinese_dataset/BZNSYP_training.csv", help='config file')
    parser.add_argument('--output_path', type=str,  default="/mnt/huanyuan2/data/speech/tts/Chinese_dataset/BZNSYP_training_prosody_length.png", help='config file')
    parser.add_argument('--ctm_file', type=str,  default="/mnt/huanyuan2/data/speech/tts/Chinese_dataset/dataset_align/BZNSYP_training/tmp/nnet3_align/ctm", help='config file')
    args = parser.parse_args()

    analysis_prosody_BZNSYP(args)


if __name__ == "__main__":
    main()

