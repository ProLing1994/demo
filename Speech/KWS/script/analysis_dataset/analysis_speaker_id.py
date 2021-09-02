import argparse
import os
import pandas as pd


def analysis_speaker_id(args):
    # load csv
    data_pd = pd.read_csv(os.path.join(args.input_dir, "total_data_files.csv"))

    # 添加 label_name, speaker_id, select_id 
    speaker_id_training_list = []
    speaker_id_validation_list = []
    for _, row in data_pd.iterrows():
        file_name = os.path.basename(row['file'])
        mode_name = os.path.basename(row['mode'])
        if "RM_KWS_ACTIVATEBWC_activatebwc_" in file_name:
            speaker_id = int(str(file_name).split("_")[-1][1:4]) 
            if mode_name == "training":
                speaker_id_training_list.append(speaker_id)
            elif mode_name == "validation":
                speaker_id_validation_list.append(speaker_id)

    speaker_id_training_list = list(set(speaker_id_training_list))
    speaker_id_validation_list = list(set(speaker_id_validation_list))
    print("speaker_num: {}/{}, total: {}".format(len(speaker_id_training_list), len(speaker_id_validation_list), len(speaker_id_training_list) + len(speaker_id_validation_list)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Streamax KWS Data Split Engine')
    args = parser.parse_args()
    # args.input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.5_03312021/"
    args.input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.5_03312021/"
    args.prefix = "RM_KWS_ACTIVATEBWC_activatebwc_"
    analysis_speaker_id(args)
    