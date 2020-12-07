import argparse
import os
import sys 

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *
from script.dataset_align.cut_keyword_wav import read_utt2wav, get_words_list


def extract_words(ctm_file, keyword_list):
    word_segments = get_words_list(ctm_file, keyword_list)
    print("word_segments:", len(word_segments))
    return word_segments


def clear_dataset(config_file):
    # load configuration file
    cfg = load_cfg_file(config_file)

    dataset_dir = cfg.general.data_dir
    csv_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date))
    ctm_file = os.path.join(dataset_dir, 'kaldi_type/tmp/nnet3_align/ctm')
    wav_file = os.path.join(dataset_dir, 'kaldi_type/wav.scp')

    if not os.path.exists(ctm_file):
        print("[Warring] No ctm file found!")
        return 

    if not "positive_label_chinese_list" in cfg.dataset.label:
        print("[Warring] No positive_label_chinese_list defined, like '小,鱼,小#,鱼#', please check!")
        return 

    keyword_list = cfg.dataset.label.positive_label_chinese_list.split(',')
    positive_label = cfg.dataset.label.positive_label[0]
    utt2wav = read_utt2wav([wav_file])
    word_segments = extract_words(ctm_file, keyword_list)

    # wav list
    wav_list = []
    for word_segment in word_segments:
        utt_id = word_segment[0]
        wav_list.append(utt2wav[utt_id])

    # update csv
    total_data_files = []                 # {'label': [], 'file': [], 'mode': []}
    csv_pd = pd.read_csv(os.path.join(csv_dir, 'total_data_files.csv'))
    for _, row in tqdm(csv_pd.iterrows()):
        if (row['label'] == positive_label and row['file'] in wav_list) or row['label'] != positive_label:
            total_data_files.append({'label': row['label'], 'file': row['file'], 'mode': row['mode']})
        else:
            tqdm.write((row['file']))

    total_data_pd = pd.DataFrame(total_data_files)
    total_data_pd.to_csv(os.path.join(csv_dir, 'total_data_files_align_clean.csv'), index=False, encoding="utf_8_sig")

def main():
    # prepare align dataset, clean the dataset according to the alignment results
    parser = argparse.ArgumentParser(description='Streamax KWS Data Split Engine')
    parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Clean the dataset according to the alignment results")
    clear_dataset(args.input)
    print("[Done] Clean the dataset")


if __name__ == "__main__":
    main()