import numpy as np
import random
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.utils.lmdb_tools import *


def dataset_augmentation_longer_senteces(cfg, text, wav, data_by_spk, lmdb_dict):
    # init
    longer_senteces_frequency = cfg.dataset.augmentation.longer_senteces_frequency 
    longer_senteces_length = cfg.dataset.augmentation.longer_senteces_length
    longer_senteces_weight = cfg.dataset.augmentation.longer_senteces_weight

    if not np.random.uniform(0, 1) < longer_senteces_frequency:
        return text, wav

    if len(data_by_spk) < 2 : 
        return text, wav

    # used to make a new version (v2) of longer senteces
    n_element = min(random.choices(
        longer_senteces_length,
        longer_senteces_weight,
        k = 1
    )[0], len(data_by_spk))
    
    data_by_spk_index = list(data_by_spk.index)
    find_data_by_spk_index = random.sample(data_by_spk_index, n_element)

    for find_idx in range(len(find_data_by_spk_index)):
        find_data_index = find_data_by_spk_index[find_idx]

        lmdb_dataset = data_by_spk.loc[find_data_index, 'dataset']
        data_name = data_by_spk.loc[find_data_index, 'unique_utterance']
        speaker = data_by_spk.loc[find_data_index, 'speaker']

        # text
        if cfg.dataset.language == 'chinese':
            find_text = data_by_spk.loc[find_data_index, cfg.dataset.symbols]
            find_text = str(find_text).strip()
        elif cfg.dataset.language == 'english':
            find_text = data_by_spk.loc[find_data_index, 'text']
        else:
            raise Exception("[ERROR:] Unknow dataset language: {}".format(cfg.dataset.language))

        # mel
        find_wav = read_audio_lmdb(lmdb_dict[lmdb_dataset], str(data_name))

        assert text[-1] in [',', '.', '/', '1', '2', '3', '4', '5'], "[ERROR:] text: {} | {} ".format(text, text[-1])
        # gen spacing
        if text[-1] == ',':
            spacing = gen_spacing(cfg.dataset.sample_rate, cfg.guided_attn.speacing_commas)
        elif text[-1] == '.':
            spacing = gen_spacing(cfg.dataset.sample_rate, cfg.guided_attn.speacing_periods)
        elif text[-1] == '/':
            spacing = gen_spacing(cfg.dataset.sample_rate, 0.0)
        else:
            text += ' '
            spacing = gen_spacing(cfg.dataset.sample_rate, 0.0)
        
        text += find_text
        wav = np.concatenate((wav, spacing, find_wav), axis=0)
    return text, wav


def gen_spacing(sr, sec) : 
    return np.zeros((int(sr * sec), ), dtype=np.float32)