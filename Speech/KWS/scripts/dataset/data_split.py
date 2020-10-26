import argparse
import copy
import glob
import hashlib
import math
import os
import pandas as pd
import random
import re
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185

def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result

def data_split(config_file, version, date):
  """ data split engine
  :param config_file:   the input configuration file
  :param version:   
  :param date:  
  :return:              None
  """
  # load configuration file
  cfg = load_cfg_file(config_file)

  # set random seed 
  random.seed(RANDOM_SEED)

  # init
  positive_label = cfg.dataset.label.positive_label
  silence_percentage = cfg.dataset.label.silence_percentage
  unknown_percentage = cfg.dataset.label.unknown_percentage
  validation_percentage = cfg.dataset.label.validation_percentage
  testing_percentage = cfg.dataset.label.testing_percentage

  all_labels_set = set()
  positive_data_files = []    # {'label': [], 'file': [], 'mode': []}
  unknown_files = []          # {'label': [], 'file': [], 'mode': []}
  silence_files = []          # {'label': [], 'file': [], 'mode': []}
  background_noise_files = [] # {'label': [], 'file': []}
  total_data_files = []       # {'label': [], 'file': [], 'mode': []}

  # Look through all the subfolders to find audio samples
  search_path = os.path.join(cfg.general.data_dir, '*', '*.wav')
  for wav_path in glob.glob(search_path):
    _, word = os.path.split(os.path.dirname(wav_path))
    word = word.lower()

    # Treat the '_background_noise_' folder as a special case, since we expect
    # it to contain long audio samples we mix in to improve training.
    if word == BACKGROUND_NOISE_DIR_NAME:
      background_noise_files.append({'label': BACKGROUND_NOISE_DIR_NAME, 'file':wav_path})
      continue
    
    all_labels_set.add(word)
    # Divide training, test and verification set
    set_index = which_set(wav_path, validation_percentage, testing_percentage)
    # If it's a known class, store its detail, otherwise add it to the list
    # we'll use to train the unknown label. 
    if word in positive_label:
      positive_data_files.append({'label': word, 'file': wav_path, 'mode':set_index})
    else:
      unknown_files.append({'label': word, 'file': wav_path, 'mode':set_index})

  if not all_labels_set:
    raise Exception('No .wavs found at ' + search_path)

  for index, wanted_word in enumerate(positive_label):
    if wanted_word not in all_labels_set:
      raise Exception('Expected to find ' + wanted_word +
                      ' in labels but only found ' +
                      ', '.join(positive_label.keys()))
  
  total_data_files.extend(positive_data_files)
  # silence and unknowns samples
  silence_wav_path = ''
  random.shuffle(unknown_files)
  for set_index in ['validation', 'testing', 'training']:
    # silence samples
    set_size = np.array([x['mode'] == set_index for x in positive_data_files]).astype(np.int).sum()
    silence_size = int(math.ceil(set_size * silence_percentage / 100))
    for _ in range(silence_size):
      silence_wav = {'label': SILENCE_LABEL, 'file': silence_wav_path, 'mode':set_index}
      silence_files.append(silence_wav)
      total_data_files.append(silence_wav)

    # unknowns samples
    unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
    unknown_files_set = [x for x in unknown_files if x['mode'] == set_index]
    unknown_files_set = copy.deepcopy(unknown_files_set)
    unknown_files_set = unknown_files_set[:unknown_size]
    for idx in range(len(unknown_files_set)):
      unknown_files_set[idx]['label'] = UNKNOWN_WORD_LABEL
    total_data_files.extend(unknown_files_set)

  # random
  random.shuffle(total_data_files)

  # output
  output_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(version, date))
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  positive_data_pd = pd.DataFrame(positive_data_files)
  unknown_pd = pd.DataFrame(unknown_files)
  silence_pd = pd.DataFrame(silence_files)
  background_noise_pd = pd.DataFrame(background_noise_files)
  total_data_pd = pd.DataFrame(total_data_files)

  positive_data_pd.to_csv(os.path.join(output_dir, 'positive_data_files.csv'), index=False)
  unknown_pd.to_csv(os.path.join(output_dir, 'unknown_files.csv'), index=False)
  silence_pd.to_csv(os.path.join(output_dir, 'silence_files.csv'), index=False)
  background_noise_pd.to_csv(os.path.join(output_dir, 'background_noise_files.csv'), index=False)
  total_data_pd.to_csv(os.path.join(output_dir, 'total_data_files.csv'), index=False)

def main():
  parser = argparse.ArgumentParser(description='Streamax KWS Data Split Engine')
  parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config.py", help='config file')
  parser.add_argument('-v', '--versions', type=str, default="1.0")
  parser.add_argument('-d', '--date', type=str, default="10162020")
  args = parser.parse_args()
  data_split(args.input, args.versions, args.date)

if __name__ == "__main__":
  main()