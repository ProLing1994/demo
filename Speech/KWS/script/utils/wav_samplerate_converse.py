import argparse
import os
import shutil

from tqdm import tqdm

def get_sub_filepaths_suffix(folder, suffix='.wav'):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            if not name.endswith(suffix):
                continue
            path = os.path.join(root, name)
            paths.append(path)
    return paths

def format_converse(args):
    # init 
    temp_path = os.path.join(args.input_dir, '{}.wav'.format('temp'))

    wave_list = get_sub_filepaths_suffix(args.input_dir)
    for idx in tqdm(range(len(wave_list))):
        audio_path = wave_list[idx]
        shutil.copy(audio_path, temp_path)
        os.system('sox {} -c 1 -b 16 -r {} -e signed-integer {}'.format(temp_path, args.samplerate, audio_path))

def main():
    parser = argparse.ArgumentParser(description="Sudio Format")
    args = parser.parse_args()
    args.input_dir = "/home/huanyuan/share/audio_data/cq_wav/xiao/"
    args.samplerate = "8k"

    format_converse(args)
    

if __name__ == "__main__":
    main()