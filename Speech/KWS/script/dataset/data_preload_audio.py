import argparse
import librosa
import multiprocessing
import pandas as pd
import pickle
import os
import sys

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from dataset.kws.dataset_helper import *
# from dataset.kws.kws_dataset import *
from utils.train_tools import load_cfg_file


def write_audio(data, output_path):
    f = open(output_path, 'wb')
    pickle.dump(data, f)
    f.close()


def multiprocessing_preload_audio(args):
    cfg, data_pd, label_index, idx = args[0], args[1], args[2], args[3]

    # init
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)

    image_name_idx = str(data_pd['file'].tolist()[idx])
    label_name_idx = str(data_pd['label'].tolist()[idx])
    mode_name_idx = str(data_pd['mode'].tolist()[idx])
    label_idx = str(label_index[label_name_idx])

    # load data
    if label_name_idx == SILENCE_LABEL:
        data = np.zeros(desired_samples, dtype=np.float32)
    else:
        data = librosa.core.load(image_name_idx, sr=sample_rate)[0]

    # output
    output_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(
        cfg.general.version, cfg.general.date), 'dataset_audio', mode_name_idx)
    output_dir_idx = os.path.join(output_dir, label_name_idx)

    if label_name_idx == SILENCE_LABEL:
        filename = label_idx + '_' + label_name_idx + '_' + str(idx) + '.txt'
    else:
        filename = label_idx + '_' + os.path.basename(os.path.dirname(
            image_name_idx)) + '_' + os.path.basename(image_name_idx).split('.')[0] + '.txt'

    write_audio(data, os.path.join(output_dir_idx, filename))
    # print("Save Results: {}".format(filename), end='\r')


def preload_audio(config_file, mode):
    """ data preprocess engine
    :param config_file:   the input configuration file
    :param mode:  
    :return:              None
    """
    print("Start preload audio({}): ".format(mode))
    # load configuration file
    cfg = load_cfg_file(config_file)

    # data index
    label_index = load_label_index(cfg.dataset.label.positive_label, cfg.dataset.label.negative_label)

    # mkdir
    output_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(
        cfg.general.version, cfg.general.date), 'dataset_audio', mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label_name_idx in label_index.keys():
        output_dir_idx = os.path.join(output_dir, label_name_idx)
        if not os.path.exists(output_dir_idx):
            os.makedirs(output_dir_idx)

    # load csv
    data_pd = pd.read_csv(cfg.general.data_csv_path)
    data_pd = data_pd[data_pd['mode'] == mode]
    data_file_list = data_pd['file'].tolist()

    # data_preload_audio
    in_params = []
    for idx in range(len(data_file_list)):
        in_args = [cfg, data_pd, label_index, idx]
        in_params.append(in_args)

    p = multiprocessing.Pool(cfg.debug.num_processing)
    out = list(tqdm(p.imap(multiprocessing_preload_audio, in_params), total=len(in_params)))
    p.close()
    p.join()
    print("Preload audio({}) Done!".format(mode))


def preload_background_audio(config_file):
    print("Start preload background audio: ")

    # load configuration file
    cfg = load_cfg_file(config_file)

    # init
    sample_rate = cfg.dataset.sample_rate

    # load csv
    background_data_pd = pd.read_csv(cfg.general.background_data_path)

    # output
    output_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(
        cfg.general.version, cfg.general.date), 'dataset_audio', BACKGROUND_NOISE_DIR_NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, row in background_data_pd.iterrows():
        background_data = librosa.core.load(row.file, sr=sample_rate)[0]
        filename = os.path.basename(row.file).split('.')[0] + '.txt'
        write_audio(background_data, os.path.join(output_dir, filename))
        print("Save Results: {}".format(filename))

    print("Preload background audio Done!")


def preload_audio_folder(args):
    config_file = args[0]
    input_dir= args[1]
    output_dir= args[2]

    print("Start preload folder audio: {}".format(os.path.basename(input_dir)))

    # load configuration file
    cfg = load_cfg_file(config_file)

    # init
    sample_rate = cfg.dataset.sample_rate

    # load data list
    data_list = os.listdir(input_dir)

    for data_idx in tqdm(range(len(data_list))):
        data_path = os.path.join(input_dir, data_list[data_idx])
        filename = os.path.basename(data_list[data_idx]).split('.')[0] + '.txt'
        if os.path.exists(os.path.join(output_dir, filename)):
            continue

        try:
            audio_data = librosa.core.load(data_path, sr=sample_rate)[0]
            write_audio(audio_data, os.path.join(output_dir, filename))
            # tqdm.write("Save Results: {}".format(filename))
            # print("Save Results: {}".format(filename), end='\r')
        except:
            continue

    print("Preload folder audio: {} Done!".format(os.path.basename(input_dir)))


def preload_augumentation_audio(config_file, speed_list, volume_list):
    # load configuration file
    cfg = load_cfg_file(config_file)

    # init 
    positive_label = cfg.dataset.label.positive_label[0]
    data_dir = cfg.general.data_dir if not cfg.general.data_dir.endswith('/') else cfg.general.data_dir[:-1]
    input_dir = os.path.join(data_dir, '../{}_augumentation'.format(os.path.basename(data_dir)))

    # mkdir
    output_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(
        cfg.general.version, cfg.general.date), 'dataset_audio', "augumentation")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    speed_list = speed_list.split(',')
    volume_list = volume_list.split(',')
    
    # multi process
    in_params = []
    for speed_idx in range(len(speed_list)):
        for volume_idx in range(len(volume_list)):
            folder_name = positive_label + "_speed_{}_volume_{}".format("_".join(speed_list[speed_idx].split('.')), "_".join(volume_list[volume_idx].split('.')))
            
            input_dir_idx = os.path.join(input_dir, folder_name)
            output_dir_idx = os.path.join(output_dir, folder_name)
            if not os.path.exists(output_dir_idx):
                os.makedirs(output_dir_idx)
            in_args = [config_file, input_dir_idx, output_dir_idx]
            in_params.append(in_args)

    p = multiprocessing.Pool(16)
    out = list(tqdm(p.imap(preload_audio_folder, in_params), total=len(in_params)))
    p.close()
    p.join()


def main():
    default_speed_list = "0.9,1.0,1.1"
    default_volume_list = "0.4,0.7,1.0,1.3,1.6"

    parser = argparse.ArgumentParser(description='Streamax KWS Data Split Engine')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config.py", help='config file')
    # parser.add_argument('-i', '--input', type=str,  default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", help='config file')
    # parser.add_argument('-i', '--input', type=str,  default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_2_label_xiaoyu.py", help='config file')
    # parser.add_argument('-i', '--input', type=str,  default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaole.py", help='config file')
    parser.add_argument('-i', '--input', type=str,  default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui.py", help='config file')
    parser.add_argument('-m', '--mode', type=str, default="training,validation,testing")
    parser.add_argument('--speed_list', type=str, default=default_speed_list)
    parser.add_argument('--volume_list', type=str, default=default_volume_list)
    args = parser.parse_args()

    print("[Begin] Data Preload")

    for mode in args.mode.split(','):
        preload_audio(args.input, mode)

    preload_background_audio(args.input)

    preload_augumentation_audio(args.input, args.speed_list, args.volume_list)
    print("[Done] Data Preload")

if __name__ == "__main__":
    main()
