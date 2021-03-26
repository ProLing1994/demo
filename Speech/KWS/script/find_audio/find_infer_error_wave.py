import argparse
import os
import pandas as pd
import shutil

def find_error_wave():
    csv_pd = pd.read_csv(args.csv_path)
    for _, row in csv_pd.iterrows():
        if row['prob_1'] <= float(args.threshold) and row['label'] == args.label:
            input_path = row['file']
            output_path = os.path.join(args.output_dir, os.path.basename(input_path).split('.')[0] + '.wav')
            print(input_path, '->', output_path)
            shutil.copy(input_path, output_path)

if __name__ == "__main__":
    default_csv_path = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_activatebwc_1_3_res15_fbankcpu_03222021/dataset_1.2_infer_longterm_validation_augmentation_False_mean.csv"
    default_output_dir = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_activatebwc_1_3_res15_fbankcpu_03222021/wrong_wave/"
    default_label = "activatebwc"
    default_threshold = '0.5'
    
    parser = argparse.ArgumentParser(description='Streamax KWS Testing Engine')
    parser.add_argument('--csv_path', type=str, default=default_csv_path)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    parser.add_argument('--label', type=str, default=default_label)
    parser.add_argument('--threshold', type=str, default=default_threshold)
    args = parser.parse_args()

    find_error_wave()
