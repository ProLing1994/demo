import argparse
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from SV.script.dataset.data_train_test_split import *


def main():
    parser = argparse.ArgumentParser(description='Streamax TTS Data Split Engine')
    parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/VC/config/cycle_vae/vc_config_cycle_vae.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Train test dataset split")
    data_split(args)
    print("[Done] Train test dataset split")


if __name__ == "__main__":
    main()
