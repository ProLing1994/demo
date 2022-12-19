import argparse
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from SV.script.dataset.data_vad import *


def main():
    # Done:
    # Chinese：BZNSYP/Aishell3
    parser = argparse.ArgumentParser(description='Streamax TTS Data Vad Engine')
    parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/tts/tts_config_english_sv2tts.py", help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/tts/tts_config_chinese_sv2tts.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Data vad")
    data_vad(args)
    print("[Done] Data vad")


if __name__ == "__main__":
    main()