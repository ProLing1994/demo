import os
import pandas as pd 

if __name__ == '__main__':
    input_dir = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_11032020/difficult_sample_mining_11122020/duplicate_audio/"
    output_dir = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_11032020/difficult_sample_mining_11122020/"
    file_list = []
    for audio_name in os.listdir(input_dir):
        file_list.append({'file':os.path.join(input_dir, audio_name)})

    file_pd = pd.DataFrame(file_list)
    file_pd.to_csv(os.path.join(output_dir, 'duplicate_audio.csv'), index=False, encoding="utf_8_sig")