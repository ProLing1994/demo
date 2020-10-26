import argparse
import multiprocessing 
import pandas as pd
import pickle
import os
import sys 

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import load_cfg_file
from dataset.kws.kws_dataset import SpeechDataset

def write_image(image, output_path):
  f = open(output_path, 'wb')
  pickle.dump(image, f)
  f.close()

def multiprocessing_data_preprocess(args):
  cfg, data_set, data_pd, idx = args[0], args[1], args[2], args[3]

  # gen model inputs
  inputs, labels, indexs = data_set[idx]

  image_idx = inputs.numpy().reshape((-1, 40))
  label_idx = str(labels.numpy())
  image_name_idx = str(data_pd['file'].tolist()[indexs])
  label_name_idx = str(data_pd['label'].tolist()[indexs])
  mode_name_idx = str(data_pd['mode'].tolist()[indexs])

  # output
  output_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date), 'dataset', mode_name_idx)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  output_dir_idx = os.path.join(output_dir, label_name_idx)
  if not os.path.exists(output_dir_idx):
    os.makedirs(output_dir_idx)

  if label_idx == '0':
    filename = label_idx + '_' + label_name_idx + '_' + str(indexs) + '.txt'
  else:
    filename = label_idx + '_' + os.path.basename(os.path.dirname(image_name_idx)) + '_' + os.path.basename(image_name_idx).split('.')[0] + '.txt'

  write_image(image_idx, os.path.join(output_dir_idx, filename))
  print("Save Results: {}".format(filename))

def data_preprocess(config_file, mode):
  """ data preprocess engine
  :param config_file:   the input configuration file
  :param mode:  
  :return:              None
  """
  print("Start data preprocess: ")
  # load configuration file
  cfg = load_cfg_file(config_file)

  # init
  data_set = SpeechDataset(cfg=cfg, mode=mode, augmentation_on=False)

  # load csv
  data_pd = pd.read_csv(cfg.general.data_csv_path)
  data_pd = data_pd[data_pd['mode'] == mode]

  # data_preprocess
  in_params = []
  for idx in tqdm(range(len(data_set))):
    in_args = [cfg, data_set, data_pd, idx]
    in_params.append(in_args)
  
  p = multiprocessing.Pool(cfg.debug.num_processing)
  out = p.map(multiprocessing_data_preprocess, in_params)
  p.close()
  p.join()
  print("Data preprocess Done!")
  
def main():
  parser = argparse.ArgumentParser(description='Streamax KWS Data Split Engine')
  parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config.py", help='config file')
  parser.add_argument('-m', '--mode', type=str, default="training")
  args = parser.parse_args()
  data_preprocess(args.input, args.mode)

if __name__ == "__main__":
  main()