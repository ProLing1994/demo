import lmdb
import os
import pandas as pd
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.utils.lmdb_tools import load_lmdb_env, read_audio_lmdb

from KWS.config.kws import hparams

def load_background_noise_lmdb(cfg):
    # load noise data
    # init 
    background_data_pd = pd.read_csv(cfg.general.background_data_path)
    background_data_lmdb_path = os.path.join(os.path.dirname(cfg.general.data_csv_path), 'dataset_audio_lmdb', '{}.lmdb'.format(hparams.BACKGROUND_NOISE_DIR_NAME))
    background_data = []

    background_data_lmdb_env = lmdb.open(background_data_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    for _, row in background_data_pd.iterrows():
        file_path = row.file
        background_data.append(read_audio_lmdb(background_data_lmdb_env, file_path))
    return background_data