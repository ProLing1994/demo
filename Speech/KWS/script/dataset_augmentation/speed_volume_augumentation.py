import argparse
import glob
import os 
import sys 

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *


def sox(args):
    volume_list = args[0]
    volume_idx = args[1]
    speed_list = args[2]
    speed_idx = args[3]
    input_path = args[4]
    output_path = args[5]
    
    if os.path.exists(output_path):
        return
    script = " ".join(['sox', '--vol', volume_list[volume_idx], input_path, output_path, 'speed', speed_list[speed_idx]])
    os.system(script)
    # tqdm.write(script)


def speed_volume_disturbution(args):
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # init 
    speed_volume_on = cfg.dataset.augmentation.speed_volume_on
    speed_list = cfg.dataset.augmentation.possitive_speed.split(',')
    volume_list = cfg.dataset.augmentation.possitive_volume.split(',')
    positive_label_list = cfg.dataset.label.positive_label
    data_dir = cfg.general.data_dir if not cfg.general.data_dir.endswith('/') else cfg.general.data_dir[:-1]

    if not speed_volume_on:
        return

    # Look through all the subfolders to find audio samples
    search_path = os.path.join(cfg.general.data_dir, '*', '*.wav')
    path_list = glob.glob(search_path)
    if 'sub_data_dir' in cfg.general:
        for sub_data_dir in cfg.general.sub_data_dir:
            sub_search_path = os.path.join(sub_data_dir, '*', '*.wav')
            path_list += glob.glob(sub_search_path)

    # check
    in_params = []
    for positive_label in positive_label_list:
        for speed_idx in range(len(speed_list)):
            for volume_idx in range(len(volume_list)):
                output_dir = os.path.join(data_dir, '../{}_augumentation'.format(os.path.basename(data_dir)), positive_label + "_speed_{}_volume_{}".format("_".join(speed_list[speed_idx].split('.')), "_".join(volume_list[volume_idx].split('.'))))

                for audio_path in path_list:
                    _, word = os.path.split(os.path.dirname(audio_path))
                    word = word.lower()
                    if word != positive_label:
                        continue

                    # mkdirs 
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    audio_name = os.path.basename(audio_path)
                    input_path = audio_path
                    output_path = os.path.join(output_dir, audio_name)

                    in_args = [volume_list, volume_idx, speed_list, speed_idx, input_path, output_path]
                    in_params.append(in_args)

        p = multiprocessing.Pool(16)
        out = list(tqdm(p.imap(sox, in_params), total=len(in_params)))
        p.close()
        p.join()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_speech.py", nargs='?', help='config file')
    # parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", nargs='?', help='config file')
    # parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaole.py", nargs='?', help='config file')
    # parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui.py", nargs='?', help='config file')
    args = parser.parse_args()

    print("[Begin] Speed Volume Augumentation")
    speed_volume_disturbution(args)
    print("[Done] Speed Volume Augumentation")

if __name__ == "__main__":
    main()