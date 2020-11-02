import argparse
import librosa
import os
import pandas as pd
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *
from dataset.kws.dataset_helper import *
from impl.pred_pyimpl import dataset_add_noise

def augmentation_test(config_file, output_dir):
    # mkdirs 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load configuration file
    cfg = load_cfg_file(config_file)

    # init 
    input_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date), 'dataset_audio')
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)

    # data index
    label_index = load_label_index(cfg.dataset.label.positive_label)

    # load data csv
    data_pd = pd.read_csv(cfg.general.data_csv_path)
    data_pd = data_pd[data_pd['mode'] == mode]
    data_file_list = data_pd['file'].tolist()
    data_mode_list = data_pd['mode'].tolist()
    data_label_list = data_pd['label'].tolist()

    for audio_index in tqdm(range(len(data_file_list))):
        if audio_index > 10:
            continue
        audio_file = data_file_list[audio_index]
        audio_mode = data_mode_list[audio_index]
        audio_label = data_label_list[audio_index]
        audio_label_idx = label_index[audio_label]

        # load data
        input_dir_index = os.path.join(input_dir, audio_mode, audio_label)
        data, filename = load_preload_audio(audio_file, audio_index, audio_label, audio_label_idx, input_dir_index)

        # alignment data
        data = np.pad(data, (0, max(0, desired_samples - len(data))), "constant")
        if len(data) > desired_samples:
            data_offset = np.random.randint(0, len(data) - desired_samples - 1)
            data = data[data_offset:(data_offset + desired_samples)]

        # add augmentation
        for time_shift_amount in time_shift_amount_list:
            time_shift_samples = int(sample_rate * time_shift_amount / 1000)
            time_shift_left = - min(0, time_shift_samples)
            time_shift_right = max(0, time_shift_samples)
            data = np.pad(data, (time_shift_left, time_shift_right), "constant")
            data = data[:len(data) - time_shift_left] if time_shift_left else data[time_shift_right:]
            data = dataset_add_noise(cfg, data)

            # output wav 
            output_path = os.path.join(output_dir, filename.split('.')[0] + '_timeshift_{}.wav'.format(str(time_shift_amount)))
            librosa.output.write_wav(output_path, data, sr=sample_rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu_2.py", nargs='?', help='config file')
    parser.add_argument('--output_dir', type=str, default="/home/huanyuan/temp/audio_augmentation")
    args = parser.parse_args()
    augmentation_test(args.input, args.output_dir)

if __name__ == "__main__":
    mode = 'training'
    time_shift_amount_list = [-1000, 1000]
    main()