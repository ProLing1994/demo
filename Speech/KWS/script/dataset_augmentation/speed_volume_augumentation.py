import argparse
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

    script = " ".join(['sox', '--vol', volume_list[volume_idx], input_path, output_path, 'speed', speed_list[speed_idx]])
    os.system(script)
    # tqdm.write(script)


def speed_volume_disturbution(config_file, speed_list, volume_list, add_data_bool=False, add_data_dir=""):
    # load configuration file
    cfg = load_cfg_file(config_file)

    # init 
    positive_label = cfg.dataset.label.positive_label[0]
    data_dir = cfg.general.data_dir if not cfg.general.data_dir.endswith('/') else cfg.general.data_dir[:-1]
    input_dir = os.path.join(data_dir, positive_label)

    # add data
    if add_data_bool:
        audio_add_list = os.listdir(add_data_dir)
    
    audio_list = os.listdir(input_dir)
    speed_list = speed_list.split(',')
    volume_list = volume_list.split(',')

    # check
    assert os.path.exists(input_dir)
    in_params = []

    for speed_idx in range(len(speed_list)):
        for volume_idx in range(len(volume_list)):
            output_dir = os.path.join(data_dir, '../{}_augumentation'.format(os.path.basename(data_dir)), positive_label + "_speed_{}_volume_{}".format("_".join(speed_list[speed_idx].split('.')), "_".join(volume_list[volume_idx].split('.'))))
            
            # mkdirs 
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for audio_name in audio_list:
                input_path = os.path.join(input_dir, audio_name)
                output_path = os.path.join(output_dir, audio_name)

                in_args = [volume_list, volume_idx, speed_list, speed_idx, input_path, output_path]
                in_params.append(in_args)

            if add_data_bool:
                for audio_name in audio_add_list:
                    input_path = os.path.join(add_data_dir, audio_name)
                    output_path = os.path.join(output_dir, audio_name)

                    in_args = [volume_list, volume_idx, speed_list, speed_idx, input_path, output_path]
                    in_params.append(in_args)

    p = multiprocessing.Pool(8)
    out = list(tqdm(p.imap(sox, in_params), total=len(in_params)))
    p.close()
    p.join()
    

def main():
    default_speed_list = "0.9,1.0,1.1"
    default_volume_list = "0.4,0.7,1.0,1.3,1.6"
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", nargs='?', help='config file')
    # parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaole.py", nargs='?', help='config file')
    parser.add_argument('--config_file', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui.py", nargs='?', help='config file')
    parser.add_argument('--speed_list', type=str, default=default_speed_list)
    parser.add_argument('--volume_list', type=str, default=default_volume_list)
    parser.add_argument('--add_data_bool', type=bool, default=False)
    parser.add_argument('--add_data_dir', type=str, default="/mnt/huanyuan/data/speech/kws/xiaoyu_dataset_11032020/xiaoyu_add_11192020/")
    args = parser.parse_args()

    print("[Begin] Speed Volume Augumentation")
    speed_volume_disturbution(args.config_file, args.speed_list, args.volume_list, args.add_data_bool, args.add_data_dir)
    print("[Done] Speed Volume Augumentation")

if __name__ == "__main__":
    main()