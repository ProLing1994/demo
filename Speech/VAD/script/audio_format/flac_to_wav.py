import argparse
import glob
import multiprocessing
import os

from tqdm import tqdm

def multiprocessing_format_transform(args):
    input_path, output_path = args[0], args[1]
    os.system("ffmpeg -i {} {}".format(input_path, output_path))


def flac_to_wav(args):
    # mkdir 
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    file_list = glob.glob(os.path.join(args.input_folder, '*/*/*' + args.input_suffix))
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

    p = multiprocessing.Pool(12)
    # out = list(tqdm(p.imap(multiprocessing_format_transform, in_params), total=len(in_params)))
    out = p.map(multiprocessing_format_transform, in_params)
    p.close()
    p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streamax Engine')
    parser.add_argument('--input_folder', type=str, default="/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech/train-clean-360/")
    parser.add_argument('--output_folder', type=str, default="/mnt/huanyuan/data/speech/asr/LibriSpeech/LibriSpeech_wav/train-clean-360/")
    parser.add_argument('--input_suffix', type=str, default=".flac")
    parser.add_argument('--output_suffix', type=str, default=".wav")
    args = parser.parse_args()

    flac_to_wav(args)