import argparse
import os
import shutil
from tqdm import tqdm


def find_audio():
    # mkdir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # init 
    wave_list = os.listdir(args.input_dir)

    for idx in tqdm(range(len(wave_list))):
        wave_path = os.path.join(args.input_dir, wave_list[idx])
        find_path = os.path.join(args.find_input_dir, wave_list[idx])

        if not os.path.exists(find_path):
            output_path = os.path.join(args.output_dir, os.path.basename(wave_path).split('.')[0] + '.wav')
            print(wave_path, '->', output_path)
            # shutil.copy(wave_path, output_path)
            shutil.move(wave_path, output_path)


if __name__ == "__main__":
    default_input_dir = "/mnt/huanyuan2/data/speech/kws/xiaoan_dataset/experimental_dataset/XiaoAnDataset/xiaoanxiaoan_8k_small_voice/"
    default_find_input_dir = "/mnt/huanyuan2/data/speech/kws/xiaoan_dataset/experimental_dataset/XiaoAnDataset/民太安数据清洗/xiaoanxiaoan_8k_small_voice/"
    default_output_dir = "/mnt/huanyuan2/data/speech/kws/xiaoan_dataset/experimental_dataset/XiaoAnDataset/民太安数据清洗/unuse/"
    
    parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--find_input_dir', type=str, default=default_find_input_dir)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    args = parser.parse_args()

    find_audio()