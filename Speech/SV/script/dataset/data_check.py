import argparse
import librosa 
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.train_tools import *
from Basic.utils.folder_tools import *


def data_check_normal(cfg, csv_path):
    # init
    drop_list = []
    desired_samples = int(cfg.dataset.sample_rate * hparams.check_wave_length_ms / 1000)

    data_pd = pd.read_csv(csv_path)
    file_list = data_pd['file'].tolist()
    
    for idx, row in tqdm(data_pd.iterrows(), total=len(file_list)):
        file_path = row['file']

        if not os.path.exists(file_path):
            drop_list.append(idx)
            continue

        # value
        data = librosa.core.load(file_path, sr=cfg.dataset.sample_rate)[0]

        # check
        if len(data) == 0 or len(data) < desired_samples:
            os.remove(file_path)
            drop_list.append(idx)
            print("remove file path: {}, wave length: {}".format(file_path, len(data)))

    print("[Information] Drop wav: {}/{}".format(len(drop_list), len(data_pd)))
    data_pd.drop(drop_list, inplace=True)
    data_pd.to_csv(csv_path, index=False, encoding="utf_8_sig")


def data_check(args):
    # load configuration file
    cfg = load_cfg_file(args.input)

    # dataset
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]
        csv_path = os.path.join(cfg.general.data_dir, dataset_name + '.csv')

        # data_check
        print("Start check dataset: {}".format(dataset_name))
        data_check_normal(cfg, csv_path)
        print("Check dataset:{}  Done!".format(dataset_name))


def main():
    # Done:
    # Chinese：SLR38/SLR68/Aishell3
    parser = argparse.ArgumentParser(description='Streamax SV Data Check Engine')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_english_TI_SV.py", help='config file')
    parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_chinese_TI_SV.py", help='config file')
    args = parser.parse_args()

    print("[Begin] Data check")
    data_check(args)
    print("[Done] Data check")


if __name__ == "__main__":
    main()