import argparse
from auditok import split
import librosa

def data_clean(input_dir, output_dir):
    audio_path = "/home/huanyuan/data/speech/kws/xiaoyu_dataset_10292020/XiaoYuDataset_10292020/xiaoyu/7278431M0_»½ÐÑ´Ê_Ð¡ÓãÐ¡Óã_ÄÐ_ÖÐÇàÄê_ÊÇ_0162.wav"

    audio_regions = split(audio_path, 2, 10, 1.5, False, True)
    idx = 0
    for region in audio_regions:
        filename = region.save(os.path.join(output_dir, "P{:0>5d}.wav".format(idx)))
        idx += 1
        print("Audio region saved as: {}".format(filename))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/home/huanyuan/data/speech/kws/xiaoyu_dataset_10292020/XiaoYuDataset_10292020/xiaoyu")
    parser.add_argument('--output_dir', type=str, default="/home/huanyuan/data/speech/kws/xiaoyu_dataset_10292020/test")
    args = parser.parse_args()
    data_clean(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()