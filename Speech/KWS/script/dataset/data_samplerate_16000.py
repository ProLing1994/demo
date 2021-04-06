import argparse
import librosa
import os 

from tqdm import tqdm

def change_type(args):
    # init 
    sample_rate = 16000
    # sample_rate = 8000

    # mkdir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_list = os.listdir(args.input_dir)
    file_list.sort()
    for file in tqdm(file_list):
        if not file.endswith(args.input_type):
            continue

        audio = librosa.core.load(os.path.join(args.input_dir, file), sr=sample_rate)[0]
        temp_path = os.path.join(args.output_dir, '{}{}'.format('temp', args.output_type))
        output_path = os.path.join(args.output_dir, '{}{}'.format(file.split('.')[0], args.output_type))
        librosa.output.write_wav(temp_path, audio, sr=sample_rate)
        os.system('sox {} -b 16 -e signed-integer {}'.format(temp_path, output_path))

def main():
    # default_input_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_04062021/xiaoanxiaoan_8k/"
    # default_output_dir = "/mnt/huanyuan/data/speech/kws/xiaoan_dataset/original_dataset/XiaoAnXiaoAn_04062021/xiaoanxiaoan_8k_wav/"
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--input_dir', type=str, default=default_input_dir)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    parser.add_argument('--input_type', type=str, default='.wav')
    parser.add_argument('--output_type', type=str, default='.wav')
    args = parser.parse_args()

    change_type(args)

if __name__ == "__main__":
    main()