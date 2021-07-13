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
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    wave_list = get_sub_filepaths_suffix(args.input_dir)
    for idx in tqdm(range(len(wave_list))):
        audio_path = wave_list[idx]

        if audio_path.endswith('_temp.wav'):
            continue

        shutil.copy(audio_path, temp_path)

        output_path = os.path.join(os.path.dirname(audio_path), os.path.splitext(os.path.basename(audio_path))[0] + '_temp.wav')
        os.system('"{}" {} -c 1 -b 16 -e signed-integer {}'.format(args.sox_path, temp_path, output_path))

def main():
    parser = argparse.ArgumentParser(description="Sudio Format")
    parser.add_argument('-i', '--input_dir', type=str, default="E:\\test\\办公室_0624")
    parser.add_argument('-s', '--sox_path', type=str, default="C:\\Program Files (x86)\\sox-14-4-2\\sox.exe")
    args = parser.parse_args()

    format_converse(args)
    

if __name__ == "__main__":
    main()