import argparse
import glob
import os
import pandas as pd
import shutil
import sys
import re

from tqdm import tqdm
                
sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *

sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.metrics_tools import get_tpr


def save_audio(output_dir, audio_path, sub_folder_name="TP"):
    output_dir = os.path.join(output_dir, sub_folder_name)

    # mkdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, os.path.basename(audio_path))
    # print(audio_path, '->', output_path)
    shutil.copy(audio_path, output_path)


def static_result(args):
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # load csv
    dataset_pd = pd.read_csv(args.csv_path)
    dataset_pd = dataset_pd[dataset_pd["type"] == args.type]
    dataset_pd = dataset_pd[dataset_pd["bool_noise_reduction"] == args.bool_noise_reduction]

    # output dir
    output_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', 
                            os.path.basename(args.csv_path).split('.')[0], 
                            args.type, 'bool_noise_reduction_' + str(args.bool_noise_reduction),
                            'result')
    # mkdir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # init
    dataset_list = []      # {'name': [], 'id':[], 'path': [], 'type': [], 'bool_noise_reduction':[], 'text':[], 'label_name':[], 'lable_number':[], 'detection_number':[]}
    fp, fn, tp = [], [], []
    
    for _, row in dataset_pd.iterrows():
        audio_name = row['name']
        audio_path = row['path']
        lable_number = row['lable_number']
        audio_dir = os.path.join(cfg.general.save_dir, 'test_straming_wav', 
                                    os.path.basename(args.csv_path).split('.')[0], args.type, 'bool_noise_reduction_' + str(args.bool_noise_reduction),
                                    audio_name.split('.')[0] + '_threshold_{}'.format('_'.join(str(cfg.test.detection_threshold).split('.'))))
        audio_list = [file for file in os.listdir(audio_dir) if file.endswith('.wav') ]
        
        # dataset_list
        dataset_list.append({'name': row['name'], 
                            'id':row['id'], 
                            'path': row['path'], 
                            'type': row['type'], 
                            'bool_noise_reduction':row['bool_noise_reduction'], 
                            'text':row['text'],
                            'label_name':row['label_name'], 
                            'lable_number':row['lable_number'],
                            'detection_number':len(audio_list)})
        
        if lable_number == 0:
            if len(audio_list) == 0:
                continue
            else:
                fp.append([len(audio_list), audio_name, audio_path])
        else:
            if len(audio_list) < lable_number:
                fn.append([lable_number - len(audio_list), audio_name, audio_path])
                tp.append([len(audio_list), audio_name, audio_path])
            elif len(audio_list) == lable_number:
                tp.append([len(audio_list), audio_name, audio_path])
            else:
                fp.append([len(audio_list) - lable_number, audio_name, audio_path])
                tp.append([lable_number, audio_name, audio_path])


    dataset_pd = pd.DataFrame(dataset_list)
    dataset_pd.to_csv(os.path.join(output_dir, 'result.csv'), index=False, encoding="utf_8_sig")
    
    print("TP: ")
    tp_number = 0
    for idx in range(len(tp)):
        tp_number += tp[idx][0]
        print(tp[idx])
    print("tp Number: {}".format(tp_number))

    print("\nFN: ")
    fn_number = 0
    for idx in range(len(fn)):
        fn_number += fn[idx][0]
        print(fn[idx])
        if bool_write_audio:
            save_audio(output_dir, fn[idx][2], "FN")
    print("FN Number: {}".format(fn_number))

    print("\nFP: ")
    fp_number = 0
    for idx in range(len(fp)):
        fp_number += fp[idx][0]
        print(fp[idx])
        if bool_write_audio:
            save_audio(output_dir, fp[idx][2], "FP")
    print("FP Number: {}".format(fp_number))

    print("\nTPR: {:.2f}%({}/{})".format(get_tpr(0, fp_number, fn_number, tp_number)*100, tp_number, tp_number + fn_number))


def main():
    # config file
    # default_config_file = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui1_6_res15_12162020/kws_config_xiaorui.py"
    default_config_file = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui1_10_res15_finetune_12162020/kws_config_xiaorui.py"
    # default_config_file = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui1_3_res15_12162020/kws_config_xiaorui.py"
    # default_config_file = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui1_11_res15_narrow_kd_12162020/kws_config_xiaorui.py"
    # default_config_file = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui1_12_res15_narrow_fintune_12162020/kws_config_xiaorui.py"
    # default_config_file = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui1_13_res15_narrow_fintune_kd_12162020/kws_config_xiaorui.py"

    # default_csv_path = "/mnt/huanyuan/data/speech/Recording_sample/Real_vehicle_sample/20201218/Real_vehicle_sample_20201218.csv"
    default_csv_path = "/mnt/huanyuan/data/speech/Recording_sample/Real_vehicle_sample/20210105/Real_vehicle_sample_20210105.csv"
    default_type = 'normal_driving'                 # ['normal_driving', 'idling_driving', '降噪前', '降噪后-SPD', '降噪后-GSC']
    default_bool_noise_reduction = False            # [False, True]

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=default_config_file)
    parser.add_argument('--csv_path', type=str, default=default_csv_path)
    parser.add_argument('--type', type=str, default=default_type)
    parser.add_argument('--bool_noise_reduction', action='store_true', default=default_bool_noise_reduction)
    args = parser.parse_args()
    static_result(args)

if __name__ == "__main__":
    bool_write_audio = True
    main()