import numpy as np
import os
import pandas as pd
import sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech')
from Basic.utils.hdf5_tools import *


def load_data_pd(cfg, mode):
    # load data_pd
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]
        csv_path = os.path.join(cfg.general.data_dir, dataset_name + '_' + mode + '_hdf5.csv')

        if not os.path.exists(csv_path):
            print("[Warning] csv path do not exist: {}".format(csv_path))
            continue

        data_pd_temp = pd.read_csv(csv_path)
        if dataset_idx == 0:
            data_pd = data_pd_temp
        else:
            data_pd = pd.concat([data_pd, data_pd_temp])

    data_pd = data_pd[data_pd["mode"] == mode]
    data_pd.reset_index(drop=True, inplace=True) 
    return data_pd


def padding(x, flen, value=0):
    """Pad values to end by flen"""
    diff = flen - x.shape[0]
    if diff > 0:
        if len(x.shape) > 1:
            x = np.concatenate([x, np.ones((diff, x.shape[1])) * value])
        else:
            x = np.concatenate([x, np.ones(diff) * value])
    return x
    

class CycleVaeDataset(Dataset):
    """
    Dataset for training mix many-to-many with speaker-posterior
    """

    def __init__(self, cfg, mode):

        self.cfg = cfg
        self.mode = mode
        self.n_cyc = self.cfg.net.yaml['cycle_vae_params']['n_cyc']
        self.stdim = self.cfg.net.yaml['cycle_vae_params']['stdim']
        self.hdf5_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_hdf5')
        self.normalize_hdf5_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_normalize_hdf5')

        self.data_pd = load_data_pd(cfg, mode)
        
        # spk
        self.spk_list = list(set(self.data_pd['speaker'].to_list()))
        self.spk_list.sort()
        self.n_spk = len(self.spk_list)
        assert self.n_spk == self.cfg.net.yaml['cycle_vae_params']['spk_dim']
        self.spk_idx_dict = {}
        for i in range(self.n_spk):
            self.spk_idx_dict[self.spk_list[i]] = i

        # key
        self.key_list = list(set(self.data_pd['key'].to_list()))
        self.key_list = [self.key_list[idx].split('.')[0] for idx in range(len(self.key_list))]
        self.key_list.sort()

        # pad
        def zero_pad(x): return padding(x, self.cfg.net.yaml['pad_len'], value=0.0)  # noqa: E704
        self.pad_transform = transforms.Compose([zero_pad])

    def __len__(self):
        return len(self.key_list)
    
    def __getitem__(self, idx):
        
        # src
        key_name_src = self.key_list[idx]
        dataset_name_src = self.data_pd.loc[idx, 'dataset']
        spk_name_src = self.data_pd.loc[idx, 'speaker']

        ## h_src
        hdf5_name_src = f"{key_name_src}.h5"
        hdf5_path_src = os.path.join(self.hdf5_dir, dataset_name_src, 'world', hdf5_name_src)
        h_src = read_hdf5(hdf5_path_src, "feat_org_lf0")                            # src 特征
        flen_src = h_src.shape[0]
        
        ## code_src class_code_src
        spk_idx = self.spk_idx_dict[spk_name_src]
        code_src = np.zeros((flen_src, self.n_spk))
        code_src[:, spk_idx] = 1
        class_code_src = np.ones(code_src.shape[0], dtype=np.int64) * spk_idx       # src spk 分类标签

        ## mean_src std_src
        hdf5_normalize_path_src = os.path.join(self.normalize_hdf5_dir, dataset_name_src, 'world', f"stats_spk_{spk_name_src}.h5")
        mean_src = read_hdf5(hdf5_normalize_path_src, "mean_feat_org_lf0")[1:2]
        std_src = read_hdf5(hdf5_normalize_path_src, "scale_feat_org_lf0")[1:2]
        
        ## spcidx_src
        spcidx_src = read_hdf5(hdf5_path_src, "spcidx_range")[0]
        flen_spc_src = spcidx_src.shape[0]

        # trg
        code_trg_list = [None] * self.n_cyc
        class_code_trg_list = [None] * self.n_cyc
        mean_trg_list = [None] * self.n_cyc
        std_trg_list = [None] * self.n_cyc
        spk_trg_list = [None] * self.n_cyc
        h_src2trg_list = [None] * self.n_cyc

        for i in range(self.n_cyc):
            ## code_trg_list
            pair_idx = np.random.randint(0, self.n_spk)
            while self.spk_list[pair_idx] == spk_name_src:
                pair_idx = np.random.randint(0, self.n_spk)
            code_trg_list[i] = np.zeros((code_src.shape[0], code_src.shape[1]))
            code_trg_list[i][:, pair_idx] = 1

            ## class_code_trg_list
            class_code_trg_list[i] = np.ones(code_src.shape[0], dtype=np.int64) * pair_idx      # trg spk 分类标签

            ## spk_trg_list
            spk_trg_list[i] = self.spk_list[pair_idx]

            ## mean_trg_list std_trg_list
            hdf5_normalize_path_trg = os.path.join(self.normalize_hdf5_dir, dataset_name_src, 'world', f"stats_spk_{self.spk_list[pair_idx]}.h5")
            mean_trg_list[i] = read_hdf5(hdf5_normalize_path_trg, "mean_feat_org_lf0")[1:2]
            std_trg_list[i] = read_hdf5(hdf5_normalize_path_trg, "scale_feat_org_lf0")[1:2]
            
            ## h_src2trg_list
            h_src2trg_list[i] = np.c_[h_src[:, :1], (std_trg_list[i]/std_src)*(h_src[:,1:2]-mean_src)+mean_trg_list[i], h_src[:, 2:self.stdim]]
            
        # torch
        h_src = torch.FloatTensor(self.pad_transform(h_src))
        spcidx_src = torch.LongTensor(self.pad_transform(spcidx_src))
        code_src = torch.FloatTensor(self.pad_transform(code_src))
        class_code_src = torch.LongTensor(self.pad_transform(class_code_src))
        for i in range(self.n_cyc):
            h_src2trg_list[i] = torch.FloatTensor(self.pad_transform(h_src2trg_list[i]))
            code_trg_list[i] = torch.FloatTensor(self.pad_transform(code_trg_list[i]))
            class_code_trg_list[i] = torch.LongTensor(self.pad_transform(class_code_trg_list[i]))

        file_src_trg_flag = False
        key_name_trg = spk_trg_list[0] + '_' + key_name_src.split('_')[1]
        hdf5_name_trg = f"{key_name_trg}.h5"
        hdf5_path_trg = os.path.join(self.hdf5_dir, dataset_name_src, 'world', hdf5_name_trg)
        if key_name_trg in self.key_list:
            file_src_trg_flag = True
            
            # h_trg 平行样本
            h_trg = read_hdf5(hdf5_path_trg, "feat_org_lf0")
            flen_trg = h_trg.shape[0]

            # spcidx_trg
            spcidx_trg = read_hdf5(hdf5_path_trg, "spcidx_range")[0]
            flen_spc_trg = spcidx_trg.shape[0]
            
            # torch
            h_trg = torch.FloatTensor(self.pad_transform(h_trg))
            spcidx_trg = torch.LongTensor(self.pad_transform(spcidx_trg))

        else:
            h_trg = h_src
            flen_trg = flen_src
            spcidx_trg = spcidx_src
            flen_spc_trg = flen_spc_src
        return {'h_src': h_src, 'flen_src': flen_src, \
                'code_src': code_src, 'class_code_src': class_code_src, \
                'spcidx_src': spcidx_src, 'flen_spc_src': flen_spc_src, \

                'h_trg': h_trg, 'flen_trg': flen_trg, \
                'code_trg_list': code_trg_list, 'class_code_trg_list': class_code_trg_list, \
                'spcidx_trg': spcidx_trg, 'flen_spc_trg': flen_spc_trg, \
                'h_src2trg_list': h_src2trg_list, \

                'spk_trg_list': spk_trg_list, \
                'file_src_trg_flag': file_src_trg_flag, \
                'hdf5_name_src': hdf5_name_src, \
                'hdf5_name_trg': hdf5_name_trg}