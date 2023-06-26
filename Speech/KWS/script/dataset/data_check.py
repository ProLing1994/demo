import argparse
import librosa
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.utils.folder_tools import *

def data_check(args):
    data_list = get_sub_filepaths_suffix(args.input_dir, '.wav')

    for idx in tqdm(range(len(data_list))):
        data_path = data_list[idx]
        
        # load data
        wave_data = librosa.core.load(data_path, sr=args.sample_rate)[0]

        if len(wave_data) <= 100:
            print(data_path)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/mnt/huanyuan2/data/speech/original/Recording/MTA_Truck_Gorila/TruckIdling/gorila_gorila/adplus1_0_70cm/")
    parser.add_argument('--sample_rate', type=int, default=8000)
    args = parser.parse_args()
    data_check(args)

if __name__ == "__main__":
    main()