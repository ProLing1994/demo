import argparse
import os 
import sys 

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *


def speed_volume_disturbution(config_file, speed_list, volume_list):
    # load configuration file
    cfg = load_cfg_file(config_file)

    # init 
    positive_label = cfg.dataset.label.positive_label[0]
    input_dir = os.path.join(cfg.general.data_dir, positive_label)
    audio_list = os.listdir(input_dir)
    speed_list = speed_list.split(',')
    volume_list = volume_list.split(',')

    # check
    assert os.path.exists(input_dir)
    for speed_idx in tqdm(range(len(speed_list))):
        for volume_idx in tqdm(range(len(volume_list))):
            output_dir = os.path.join(cfg.general.data_dir, positive_label + "_speed_{}_volume_{}".format("_".join(speed_list[speed_idx].split('.')), "_".join(volume_list[volume_idx].split('.'))))
            
            # mkdirs 
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for audio_name in tqdm(audio_list):
                input_path = os.path.join(input_dir, audio_name)
                output_path = os.path.join(output_dir, audio_name)
                script = " ".join(['sox', '--vol', volume_list[volume_idx], input_path, output_path, 'speed', speed_list[speed_idx]])
                # print(script)
                os.system(script)
    

def main():
    default_speed_list = "0.9,1.0,1.1"
    default_volume_list = "0.4,0.7,1.0,1.3,1.6"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu_2.py", nargs='?', help='config file')
    parser.add_argument('--speed_list', type=str, default=default_speed_list)
    parser.add_argument('--volume_list', type=str, default=default_volume_list)
    args = parser.parse_args()
    speed_volume_disturbution(args.config_file, args.speed_list, args.volume_list)


if __name__ == "__main__":
    main()