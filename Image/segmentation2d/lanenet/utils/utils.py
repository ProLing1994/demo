import os
import torch
import shutil
import numpy as np
from random import seed, shuffle
from thop import profile, clever_format

def readlist(filepath):
    datalist = []
    with open(filepath, 'r+') as f:
        for i in f:
            datalist.append(i.strip())
    return datalist

def writelist(datalist, listpath):
    with open(listpath, 'w+') as f:
        for i in datalist:
            f.write(i)
            f.write('\n')
    return 0

def buildDir(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

def rebuildDir(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    else:
        shutil.rmtree(filepath)
        os.makedirs(filepath)

def trainval_split(datalist, tran_rate=0.9, rand_seed=10):
    seed(rand_seed)
    shuffle(datalist)
    N = len(datalist)
    split = int(N*tran_rate)
    trainlist = datalist[: split]
    validlist = datalist[split: ]
    return trainlist, validlist

def OPcounter(model, channel, input_shape, device, verbose=True):
    from torch.autograd import Variable
    input_shape = (channel, *input_shape)
    input = Variable(torch.randn(1, *input_shape)).to(device)
    flops, params = profile(model, inputs=(input, ), verbose=verbose)
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params

def get_filelist(path, typefile=['.png']):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    Filelist = [i.replace('\\', '/') for i in Filelist]
    Filelist = [i for i in Filelist if os.path.splitext(i)[-1] in typefile]
    return Filelist
