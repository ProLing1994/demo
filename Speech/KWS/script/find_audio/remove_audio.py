import argparse
import os
from tqdm import tqdm

def remove_audio():
    wave_list = os.listdir(args.input_dir)

    for idx in tqdm(range(len(wave_list))):
        wave_path = os.path.join(args.input_dir, wave_list[idx])
        output_path = os.path.join(args.output_dir, os.path.basename(wave_path).split('.')[0] + '.wav')

        if os.path.exists(output_path):
            print(output_path)
            # os.remove(output_path)
    
if __name__ == "__main__":
    default_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/experimental_dataset/XiaoAnDataset/xiaoanxiaoan_16k_small_voice/"
    default_output_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/experimental_dataset/XiaoAnDataset/xiaoanxiaoan_16k/"
    
    parser = argparse.ArgumentParser(description='Streamax KWS Engine')
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    args = parser.parse_args()

    remove_audio()