# -*- coding: utf-8 -*-

import os
import torch
import argparse
from net import SegNet
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def parse_args():
    parser = argparse.ArgumentParser()
    # phase
    parser.add_argument('--phase'     , type=str  , default='train'  , choices=['train', 'test', 'inference'])
    # Hyperparameters
    parser.add_argument('--batchsize' , type=int  , default=256      , help='train batchsize')
    parser.add_argument('--lr'        , type=float, default=1e-3     , help='learning rate')
    parser.add_argument('--lr_decay'  , type=int  , default=100      , help='epoch update for SGD')
    parser.add_argument('--lr_gamma'  , type=float, default=0.5      , help='Gamma update for SGD')
    parser.add_argument('--workers'   , type=int  , default=20       , help='num of cpu cores')
    parser.add_argument('--max_epoch' , type=int  , default=500      , help='train epochs')
    parser.add_argument('--patience'  , type=int  , default=24       , help='early stop patience')
    parser.add_argument('--resume'    , type=str  , default='best'   , help=' `` or `best` or `epoch123` ')
    parser.add_argument('--device'    , type=str  , default='cuda:1' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--onlinetest', type=str  , default=True     , help='compute metrics when training')
    parser.add_argument('--augrate'   , type=float, default=0.4      , help='rate of aug')
    parser.add_argument('--labelfile' , type=str  , default='json'   , choices=['mask','json'])
    # project dirs (Relative path)
    parser.add_argument('--rootdir'   , type=str  , default= '/lirui/Projects/R151Seg/')
    parser.add_argument('--weightdir' , type=str  , default='weights', help='dir for saving models')
    parser.add_argument('--logdir'    , type=str  , default='logs'   , help='dir for saving training logs')
    parser.add_argument('--results'   , type=str  , default='results', help='dir for saving infer samples')
    # model parameter
    parser.add_argument('--labelmode' , type=str  , default='single'  , choices=['single','multi'])
    parser.add_argument('--loss'      , type=str  , default='dcfcce'    , choices=['ce', 'focal', 'dcce', 'dcfc', 'dcfcce', 'rmitopk'])
    parser.add_argument('--fc'        , type=int  , default= 32)
    parser.add_argument('--nblock'    , type=int  , default= 1)
    parser.add_argument('--softmax'   , type=bool , default= False)
    parser.add_argument('--downsample_mode', type=str  , default= 'conv')
    parser.add_argument('--upsample_mode'  , type=str  , default= 'conv')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    net = SegNet(args)
    net.train_net()
    
    # net.test_net('best')
    # net.inference('results/inputs/SC', 'preview', 'results/outputs/SC')
    # net.inference_video('/lirui/TestSamples/C53/WA', 'save', 'test/videos', epoch=200, jump=5, )
    
    # for i in [3,5,7,9,11,13]: # 10, 15, 20, 25, 35
    #     net.test_net(i, False)



# nohup python main.py --resume epoch75 --lr_decay 30 &




