import argparse
import librosa
import multiprocessing 
import pandas as pd
import pickle
import os
import sys 

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import load_cfg_file
from dataset.kws.kws_dataset import *
from dataset.kws.dataset_helper import *


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
  output_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date), 'dataset_audio', mode_name_idx)
  output_dir_idx = os.path.join(output_dir, label_name_idx)

  if label_idx == '0':
    filename = label_idx + '_' + label_name_idx + '_' + str(idx) + '.txt'
  else:
    filename = label_idx + '_' + os.path.basename(os.path.dirname(image_name_idx)) + '_' + os.path.basename(image_name_idx).split('.')[0] + '.txt'

  write_audio(data, os.path.join(output_dir_idx, filename))
  print("Save Results: {}".format(filename))


def preload_audio(config_file, mode):
  """ data preprocess engine
  :param config_file:   the input configuration file
  :param mode:  
  :return:              None
  """
  print("Start preload audio: ")
  # load configuration file
  cfg = load_cfg_file(config_file)

  # data index
  label_index = load_label_index(cfg.dataset.label.positive_label)

  # mkdir
  output_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date), 'dataset_audio', mode)
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
  for idx in tqdm(range(len(data_file_list))):
    in_args = [cfg, data_pd, label_index, idx]
    in_params.append(in_args)
  
  p = multiprocessing.Pool(cfg.debug.num_processing)
  out = p.map(multiprocessing_preload_audio, in_params)
  p.close()
  p.join()
  print("Preload audio Done!")
  

def preload_background_audio(config_file):
  print("Start preload background audio: ")

  # load configuration file
  cfg = load_cfg_file(config_file)

  # init 
  sample_rate = cfg.dataset.sample_rate

  # load csv
  background_data_pd = pd.read_csv(cfg.general.background_data_path)

  # output
  output_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(cfg.general.version, cfg.general.date), 'dataset_audio', BACKGROUND_NOISE_DIR_NAME)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  for idx, row in background_data_pd.iterrows():
    background_data = librosa.core.load(row.file, sr=sample_rate)[0]
    filename = os.path.basename(row.file).split('.')[0] + '.txt'
    write_audio(background_data, os.path.join(output_dir, filename))
    print("Save Results: {}".format(filename))

  print("Preload background audio Done!")


def main():
  parser = argparse.ArgumentParser(description='Streamax KWS Data Split Engine')
  # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config.py", help='config file')
  # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", help='config file')
  parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu_2.py", help='config file')
  # parser.add_argument('-m', '--mode', type=str, default="training")
  # parser.add_argument('-m', '--mode', type=str, default="validation")
  parser.add_argument('-m', '--mode', type=str, default="testing")
  args = parser.parse_args()
  preload_audio(args.input, args.mode)
  # preload_background_audio(args.input)


if __name__ == "__main__":
  main()