import argparse
import glob
import os 

parser = argparse.ArgumentParser(description="Audio Split Using Auditok")
parser.add_argument('--input_dir', type=str, default="/mnt/huanyuan/data/speech/Recording_sample/MKV_movie_sample/original_dataset/")
parser.add_argument('--output_dir', type=str, default="/mnt/huanyuan/data/speech/Recording_sample/MKV_movie_sample/result/")
parser.add_argument('--file_type', type=str, default=".wav")
parser.add_argument('--speaker', type=int, default=1)
parser.add_argument('--idx', type=int, default=1)
args = parser.parse_args()


def main():
    speaker = args.speaker
    file_list = glob.glob(os.path.join(args.input_dir, '*/*' + args.file_type))
    file_list.sort()

    for idx in range(len(file_list)):
        file_path = file_list[idx]
        os.system("python/home/huanyuan/code/demo/Speech/VAD/script/audio_split_rename/audio_split.py --audio_path {} --output_dir {} --speaker {} --idx {}".format(
                    file_path, args.output_dir, speaker, args.idx))
        speaker += 1;
    

if __name__ == "__main__":
    main()
