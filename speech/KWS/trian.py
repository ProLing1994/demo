import argparse
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo/speech/KWS')
from utils.train_tools import *

sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.logging_helpers import setup_logger

def train(config_file):
  """ training engine
  :param config_file:   the input configuration file
  :return:              None
  """
  # load configuration file
  cfg = load_cfg_file(config_file)
  print()

  # clean the existing folder if the user want to train from scratch
  setup_workshop(cfg)

  # control randomness during training
  init_torch_and_numpy(cfg)

  # enable logging
  log_file = os.path.join(cfg.general.save_dir, 'logging', 'train_log.txt')
  logger = setup_logger(log_file, 'kws')

  # define network
  net = import_network(cfg)

  # define loss function
  loss_func = define_loss_function(cfg)

  # load checkpoint if resume epoch > 0
  if cfg.general.resume_epoch >= 0:
    last_save_epoch, batch_idx = load_checkpoint(cfg.general.resume_epoch, net,
                                                  cfg.general.save_dir)
    start_epoch = last_save_epoch
  else:
    start_epoch, last_save_epoch, batch_idx = 0, 0, 0

  # set training optimizer, learning rate scheduler
  optimizer = set_optimizer(cfg, net)

  # get training data set and test data set
  train_dataloader, len_dataset = generate_training_dataset(cfg)
  # if cfg.general.is_test:
  #   eval_train_dataloader, eval_test_dataloader = generate_testing_data_set(cfg)

def main():
  parser = argparse.ArgumentParser(description='Streamax KWS Training Engine')
  parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/speech/KWS/config/kws/kws_config.py", nargs='?', help='config file')
  args = parser.parse_args()
  train(args.input)

if __name__ == "__main__":
  main()