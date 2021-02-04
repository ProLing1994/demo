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


def clear_dataset(args):
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # dataset_dir_list
    dataset_dir_list = []
    dataset_dir_list.append(cfg.general.data_dir)
    dataset_dir_list.extend(cfg.general.sub_data_dir)
    positive_label_list = cfg.dataset.label.positive_label
    csv_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date))

    wav_list = []
    for label_idx in range(len(cfg.dataset.label.positive_label)):
        for data_dir in dataset_dir_list:
            positive_label = cfg.dataset.label.positive_label[label_idx]
            ctm_file = os.path.join(data_dir, 'kaldi_type/{}/tmp/nnet3_align/ctm'.format(positive_label))
            wav_file = os.path.join(data_dir, 'kaldi_type/{}/wav.scp'.format(positive_label))

            if not os.path.exists(ctm_file):
                continue

            if not "positive_label_chinese_name_list" in cfg.dataset.label:
                print("[Warring] No positive_label_chinese_name_list defined, like '小,鱼,小#,鱼#', please check!")
                return 

            positive_label_chinese_name = cfg.dataset.label.positive_label_chinese_name_list[label_idx]
            keyword_list = positive_label_chinese_name.split(',')
            print("label: {}, positive_label_chinese_name:{}".format(positive_label, positive_label_chinese_name))

            utt2wav = read_utt2wav([wav_file])
            word_segments = extract_words(ctm_file, keyword_list)
            
            # wav list
            for word_segment in word_segments:
                utt_id = word_segment[0]
                wav_list.append(utt2wav[utt_id])

    if not wav_list:
        return
        
    # update csv
    total_data_files = []                 # {'label': [], 'file': [], 'mode': []}
    csv_pd = pd.read_csv(os.path.join(csv_dir, 'total_data_files.csv'))
    clean_data_num = 0
    for _, row in tqdm(csv_pd.iterrows()):
        if (row['label'] in positive_label_list and row['file'] in wav_list) or row['label'] not in  positive_label_list:
            total_data_files.append({'label': row['label'], 'file': row['file'], 'mode': row['mode']})
        else:
            clean_data_num += 1
            tqdm.write("clean num: {}, label: {}, file: {}".format(clean_data_num, row['label'], row['file']))

    total_data_pd = pd.DataFrame(total_data_files)
    total_data_pd.to_csv(os.path.join(csv_dir, 'total_data_files_align_clean.csv'), index=False, encoding="utf_8_sig")

def main():
    # prepare align dataset, clean the dataset according to the alignment results
    parser = argparse.ArgumentParser(description='Streamax KWS Data Split Engine')
    # parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", help='config file')
    parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_pretrain.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Clean the dataset according to the alignment results")
    clear_dataset(args)
    print("[Done] Clean the dataset")


if __name__ == "__main__":
    main()
