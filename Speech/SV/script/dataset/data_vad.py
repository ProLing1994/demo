import argparse
import multiprocessing 
import sys 
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio
from Basic.utils.train_tools import *
from Basic.utils.folder_tools import *


def data_vad_multiprocessing(in_args):
    cfg = in_args[0]
    data_path = in_args[1]
    wav = audio.preprocess_wav(data_path, cfg.dataset.sample_rate)
    sf.write(data_path, wav, cfg.dataset.sample_rate)
    print("Done: {}".format(data_path))


def data_vad_normal(cfg, dataset_name):
    '''
    data_vad_normal
    数据格式：
        不重要
    说明：
        对文件夹下所有音频数据执行 vad 操作，同时替换原有数据
    '''
    dataset_path = cfg.general.TISV_dataset_path_dict[dataset_name]

    in_params = []
    data_list = get_sub_filepaths_suffix(dataset_path)
    data_list.sort()
    for idx in tqdm(range(len(data_list))):
        data_path = data_list[idx]
        in_args = [cfg, data_path]
        in_params.append(in_args)

    p = multiprocessing.Pool(8)
    list(tqdm(p.imap(data_vad_multiprocessing, in_params), total=len(in_params)))
    # data_vad_multiprocessing(in_params[0])


def data_vad(args):
    # load configuration file
    cfg = load_cfg_file(args.input)

    # dataset
    for dataset_idx in range(len(cfg.general.TISV_dataset_list)):
        dataset_name = cfg.general.TISV_dataset_list[dataset_idx]

        data_vad_normal(cfg, dataset_name)


def main():
    # Done:
    # Chinese：SLR38/SLR68/Aishell3
    parser = argparse.ArgumentParser(description='Streamax SV Data Split Engine')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_english_TI_SV.py", help='config file')
    parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_chinese_TI_SV.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Train test dataset split")
    data_vad(args)
    print("[Done] Train test dataset split")


if __name__ == "__main__":
    main()