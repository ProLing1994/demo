import argparse
import os
import sys
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.train_tools import load_cfg_file
from Basic.utils.folder_tools import *
from Basic.utils.hdf5_tools import *

# from VOCODER.dataset.vocoder.vocoder_wavegan_dataset_preload_audio_lmdb import VocoderWaveGanDataset
from VOCODER.dataset.vocoder.vocoder_wavegan_dataset_preload_audio_hdf5 import VocoderWaveGanDataset


def general_normalize_state(args, cfg, scaler, mode_type='testing'):

    # define dataset
    # 目前仅用于 WaveGanDataset
    dataset = VocoderWaveGanDataset(cfg, mode_type)
    print("The number of files: {}".format(len(dataset)))

    for wav, mel in tqdm(dataset):
        scaler.partial_fit(mel)


def data_normalize(args):
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # mkdir
    # output_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_normalize_lmdb')
    output_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_normalize_hdf5')
    create_folder(output_dir)

    # calculate statistics
    scaler = StandardScaler()

    # mode 
    for mode_idx in range(len([hparams.TRAINING_NAME, hparams.TESTING_NAME, hparams.VALIDATION_NAME])):

        mode_type = [hparams.TRAINING_NAME, hparams.TESTING_NAME, hparams.VALIDATION_NAME][mode_idx]

        # check
        dataset_path = [cfg.general.dataset_path_dict[dataset_name + "_" + mode_type] 
                if dataset_name + "_" + mode_type in cfg.general.dataset_path_dict else '' 
                for dataset_name in cfg.general.dataset_list]
        find_dataset_path = np.array([dataset_path[idx] != '' for idx in range(len(dataset_path))]).any()
        if not find_dataset_path:
            print("[Warning] Do not find dataset mode_type: {}".format(mode_type))
            continue
        
        # generate 
        print("Start normalize dataset mode_type: {}".format(mode_type))
        general_normalize_state(args, cfg, scaler, mode_type=mode_type) 
        print("Normalize dataset mode_type:{} Done!".format(mode_type))

    # save statistics
    dataset_name = '_'.join(cfg.general.dataset_list)
    write_hdf5(
        os.path.join(output_dir, dataset_name + "_stats.h5"),
        "mean",
        scaler.mean_.astype(np.float32),
    )
    write_hdf5(
        os.path.join(output_dir, dataset_name + "_stats.h5"),
        "scale",
        scaler.scale_.astype(np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description='Streamax SV Data Split Engine')
    parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/TTS/config/tts/tts_config_chinese_sv2tts.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Data Normalize")
    data_normalize(args)
    print("[Done] Data Normalize")


if __name__ == "__main__":
    main()