import argparse
import glob
import os
import pandas as pd
import shutil
import re

from tqdm import tqdm

def find_audio(args):
    # init
    dataset_list = []      # {'name': [], 'id':[], 'path': [], 'type': [], 'bool_noise_reduction':[], 'text':[], 'label_name':[], 'lable_number':[]}

    # find txt
    suffix = '.txt'
    txt_list = glob.glob(os.path.join(args.input_dir, '*/*/*/' + '*' + suffix))

    for idx in tqdm(range(len(txt_list))):
        txt_path = txt_list[idx]
        dir_path = os.path.dirname(txt_path)
        tqdm.write("Do: {}".format(txt_path))

        with open(txt_path, "r", encoding="gbk") as f:
            lines = f.readlines()
            for line in lines:
                # check
                if not line.strip():
                    continue
                assert len(line.strip().split()) == 2
                
                # load id and text
                audio_id = int(line.strip().split()[0])
                audio_text = line.strip().split()[1]
                audio_text = audio_text.replace(',', '')
                audio_text = audio_text.replace('，', '')
                audio_text = audio_text.replace('.', '')
                audio_text = audio_text.replace('。', '')

                # find audio
                suffix = '.wav'
                audio_list = glob.glob(os.path.join(dir_path, '*P{:0>5d}'.format(audio_id) + suffix))

                if len(audio_list) != 1:
                    print("[Warring] audio:{} do not exist, please check!".format(os.path.join(dir_path, '*P{:0>5d}'.format(audio_id) + suffix)))
                    continue

                audio_path = audio_list[0]
                audio_type = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(audio_path))))
                bool_noise_reduction = True if os.path.basename(os.path.dirname(os.path.dirname(audio_path))) == 'after_noise_reduction' else False
                label_list = re.findall(args.chinese_names, audio_text)
                label_name = None if not len(label_list) else args.label_names
                lable_number = len(label_list)

                # dataset_list
                dataset_list.append({'name': os.path.basename(audio_path), 
                                    'id':audio_id, 
                                    'path': audio_path, 
                                    'type': audio_type, 
                                    'bool_noise_reduction':bool_noise_reduction, 
                                    'text':audio_text,
                                    'label_name':label_name, 
                                    'lable_number':lable_number})
    dataset_pd = pd.DataFrame(dataset_list)
    dataset_pd.to_csv(os.path.join(args.input_dir, os.path.basename(os.path.dirname(args.input_dir)) + '_' + os.path.basename(args.input_dir) + '.csv'), index=False, encoding="utf_8_sig")
                

def main():
    parser = argparse.ArgumentParser(description="Prepare Real Vehicle Sample Dataset")
    parser.add_argument('--input_dir', type=str, default="/mnt/huanyuan/data/speech/Real_vehicle_sample/20201218")
    parser.add_argument('--audio_format', type=str, default="RM_Carchat_Mandarin_S{:0>3d}P{:0>5d}.wav")
    parser.add_argument('--label_names', type=str, default='xiaorui')
    parser.add_argument('--chinese_names', type=str, default='小锐小锐')
    args = parser.parse_args()

    find_audio(args)

if __name__ == "__main__":
    main()