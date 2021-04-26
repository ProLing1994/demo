import argparse
import os
import pandas as pd


def audio_lable_csv(args):
    file_list = os.listdir(args.input_folder)
    file_list.sort()

    for idx in range(len(file_list)):
        if not file_list[idx].endswith('.txt'):
            continue
        
        file_path = os.path.join(args.input_folder, file_list[idx])
        label_csv_list = [] # {'file': ..., 'label': ..., 'start_time': ..., 'end_time': ...}
        with open(file_path, "r") as f :
            lines = f.readlines()
            
            for line in lines:
                label_csv_dict = {}
                label_csv_dict['file'] = file_list[idx]
                label_csv_dict['label'] = str(line.strip().split(':')[-1].replace(' ', ''))
                label_csv_dict['start_time'] = int(line.strip().split(':')[0].split('~')[0]) * 1000.0 / args.sample_rate
                label_csv_dict['end_time'] = int(line.strip().split(':')[0].split('~')[1]) * 1000.0 / args.sample_rate
                label_csv_list.append(label_csv_dict)

        label_csv_pd = pd.DataFrame(label_csv_list)
        label_csv_pd.to_csv(file_path.split('.')[0] + '.csv', index=False, encoding="utf_8_sig")
        # label_csv_pd.to_csv(file_path.split('.')[0].replace('_asr_', '_ori_') + '.csv', index=False, encoding="utf_8_sig")
        # label_csv_pd.to_csv(file_path.split('.')[0].replace('_ori_', '_asr_') + '.csv', index=False, encoding="utf_8_sig")

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/kws/english_kws_dataset/test_dataset/海外同事录制_0425/办公室场景/场景二/") 
    args = parser.parse_args()

    # params
    args.sample_rate = 16000

    audio_lable_csv(args)


if __name__ == "__main__":
    main()