import argparse
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.folder_tools import *
from Basic.utils.train_tools import *

from VC.network.cycle_vae.cycle_vae import CycleVae
from VC.utils.cycle_vae.train_tools import generate_dataset_cycle_vae, generate_test_dataset_cycle_vae


def train(args):
    """ training engine
    :param config_file:   the input configuration file
    :return:              None
    """
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # clean the existing folder if the user want to train from scratch
    setup_workshop(cfg)
    
    # copy config file
    copy_config_file(cfg, args.config_file)

    # control randomness during training
    init_torch_and_numpy(cfg)

    # define network
    network = CycleVae(cfg)

    # define dataset
    train_dataloader, len_train_dataset = generate_dataset_cycle_vae(cfg, hparams.TRAINING_NAME)
    eval_dataloader, len_eval_dataset = generate_test_dataset_cycle_vae(cfg, hparams.TESTING_NAME)

    network.train(train_dataloader, len_train_dataset, eval_dataloader, len_eval_dataset)

    return

def main(): 
    parser = argparse.ArgumentParser(description='Streamax VC Training Engine')
    parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/VC/config/cycle_vae/vc_config_cycle_vae.py", nargs='?', help='config file')
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()