import argparse
import os
import pandas as pd


def analysis_speaker_id(args):
    # load csv
    data_pd = pd.read_csv(os.path.join(args.input_dir, "total_data_files.csv"))

    # 添加 label_name, speaker_id, select_id 
    training_list = []
    validation_list = []
    for _, row in data_pd.iterrows():
        file_name = os.path.basename(row['file'])
        mode_name = os.path.basename(row['mode'])
        if args.prefix in file_name:
            # speaker_id = int(str(file_name).split("_")[-1][1:4]) 
            speaker_id = int(str(file_name).split("_")[-1][1:7]) 
            if mode_name == "training":
                training_list.append(speaker_id)
            elif mode_name == "validation":
                validation_list.append(speaker_id)

    training_num = len(training_list)
    validation_num = len(validation_list)
    training_list = list(set(training_list))
    validation_list = list(set(validation_list))
    print("speaker_num: {}/{}({}/{}), total: {}({})".format(len(training_list), len(validation_list), training_num, validation_num, len(training_list) + len(validation_list), training_num + validation_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Streamax KWS Data Split Engine')
    args = parser.parse_args()
    # args.input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.5_03312021/"
    # args.input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_2s_1.7_08192021/"
    # args.input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_tts_2s_1.8_08272021/"
    args.input_dir = "/mnt/huanyuan/data/speech/kws/english_kws_dataset/experimental_dataset/dataset_activatebwc_sztts_2s_1.10_08302021/"
    # args.prefix = "RM_KWS_ACTIVATEBWC_activatebwc_"
    args.prefix = "RM_KWS_ACTIVATEBWC_TTSsv2tts_"
    analysis_speaker_id(args)
    