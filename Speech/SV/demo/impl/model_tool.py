import importlib
import numpy as np
import os
import sys

import torch

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio
from SV.utils.infer_tools import embedding
from SV.utils.visualizations_tools import *

def pytorch_sv_model_init(cfg, chk_file, model_name, class_name, use_gpu=False):
    # init model
    os.sys.path.insert(0, os.path.dirname(model_name))
    net_module = importlib.import_module(os.path.splitext(os.path.basename(model_name))[0])
    os.sys.path.pop(0)
    
    # load model
    net = net_module.__getattribute__(class_name)(cfg)
    
    if use_gpu:
        net = net.cuda()

    # load state
    state = torch.load(chk_file)
    new_state = {}
    for k,v in state['state_dict'].items():
        name = k[7:]
        new_state[name] = v
    net.load_state_dict(new_state)

    net.eval()
    return net

def pytorch_sv_model_forward(cfg, net, wav, use_gpu=False):
    del use_gpu
    
    # int 2 float
    wav = wav.astype(np.float32)
    wav = wav / float(pow(2, 15))

    # preprocess_wav
    wav = audio.preprocess_wav(wav, cfg.dataset.sample_rate)
    # audio.save_wav(wav, "/home/huanyuan/temp/data.wav", cfg.dataset.sample_rate)
    
    # embedding
    embed = embedding(cfg, net, wav)
    return embed

def show_embedding(embeds_list):
    end_embedding = embeds_list[-1]
    
    dist_list = []
    for idx in range(len(embeds_list) - 1):
        idx_embeddig = embeds_list[idx]
        similiarity = np.dot(idx_embeddig, end_embedding.T)
        dist = 1. - similiarity
        dist_list.append(dist)
    
    print(dist_list)

    # embeds = np.array(embeds_list)
    # draw_projections_speaker(embeds, out_fpath="/home/huanyuan/temp/embedding.jpg")