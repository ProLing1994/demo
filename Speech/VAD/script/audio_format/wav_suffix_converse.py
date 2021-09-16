import argparse
import multiprocessing
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

def multiprocessing_format_transform(args):
    input_path, output_path = args[0], args[1]
    if args.bool_windows:
        os.system('"{}" {} -c 1 -b 16 -e signed-integer {}'.format(args.sox_path, input_path, output_path))
    else:
        os.system("ffmpeg -loglevel panic -i {} {} ".format(input_path, output_path))
        os.system("rm {}".format(input_path))

def wav_suffix_converse(args):
    # mkdir 
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    file_list = get_sub_filepaths_suffix(args.input_folder, args.input_suffix)
    file_list.sort()
    print(len(file_list))

    in_params = []
    for idx in tqdm(range(len(file_list))):
        input_path = file_list[idx]

        if not input_path.endswith(args.input_suffix):
            continue

        output_path = input_path.replace(args.input_folder, args.output_folder)
        output_path = output_path.replace(args.input_suffix, args.output_suffix)
        tqdm.write("{} -> {}".format(input_path, output_path))

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        in_args = [input_path, output_path]
        in_params.append(in_args)

    p = multiprocessing.Pool(16)
    out = list(tqdm(p.imap(multiprocessing_format_transform, in_params), total=len(in_params)))
    p.close()
    p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streamax Engine')
    parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/sv/VoxCeleb2/dev/")
    parser.add_argument('--output_folder', type=str, default="/mnt/huanyuan/data/speech/sv/VoxCeleb2/dev_wav/")
    parser.add_argument('--input_suffix', type=str, default=".m4a")
    parser.add_argument('--output_suffix', type=str, default=".wav")
    parser.add_argument('-w', '--bool_windows', type=bool, default=False)
    parser.add_argument('-s', '--sox_path', type=str, default="C:\\Program Files (x86)\\sox-14-4-2\\sox.exe")
    args = parser.parse_args()

    wav_suffix_converse(args)