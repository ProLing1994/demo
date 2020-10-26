import argparse
import sys


def main():

    default_gpu_id = 0
    default_data_csv_path = "/home/huanyuan/data/speech/kws/tf_speech_commands/dataset_1.0_10162020/test.csv"
    default_background_data_path = "/home/huanyuan/data/speech/kws/tf_speech_commands/dataset_1.0_10162020/background_noise_files.csv"
    default_mode = "validation"


    parser = argparse.ArgumentParser(description='Streamax KWS Infering Engine')
    parser.add_argument('--gpu_id', type=int, default=default_gpu_id, help='the gpu id to run model')
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
