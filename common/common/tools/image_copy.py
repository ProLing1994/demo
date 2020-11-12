import glob
import os
import pandas as pd
import shutil

if __name__ == '__main__':
    input_csv = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaoyu_mining_difficult_sample_11122020/infer_difficult_sample_mining_11112020.csv"
    input_dir = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_11032020/difficult_sample_mining_11122020/audio/"
    output_dir = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_11032020/difficult_sample_mining_11122020/clean_audio/"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    files_pd = pd.read_csv(input_csv)
    for _, row in files_pd.iterrows():
        if row['prob_1'] <= 0.9:
            input_path = os.path.join(input_dir, os.path.basename(row['file']).split('.')[0] + '.wav')
            output_path = os.path.join(output_dir, os.path.basename(row['file']).split('.')[0] + '.wav')
            print(input_path, '->', output_path)
            shutil.copy(input_path, output_path)
