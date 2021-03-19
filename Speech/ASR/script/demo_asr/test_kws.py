#from model_language.LanguageModel import ModelLanguage
#coding:utf-8
# -*- coding: utf-8 -*-
import os
from get_mfcc import GetMfscFeature
import numpy as np
import sys

import soundfile as sf
import torch

caffe_root = '/home/huanyuan/code/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

kws_list=['start recording','stop recording','mute audio','unmute aduio','shots fired',
          'freeze','drop gun','keep hand','get down on the ground']

kws_listall=['start record','stop record','mute audio','unmute aduio',
             'shot fire','shot fire','shot fire',
             'freeze','freeze','freeze',
             'drop gun','drop gun','drop gun','drop gun',
             'keep hand','put hand','put hand','put hand','keep hand','keep hand',
             'down ground','down ground','down ground','down ground','down ground']
def kws_detect(feature,net,transformer):
    x_in = feature.reshape(feature.shape[0], feature.shape[1], 1)
    net.blobs['data'].data[...] = transformer.preprocess('data', x_in)
    out = net.forward()
    out=out['conv_blob39'][0]
    out=np.transpose(out,(2,1,0))
    return out

def torch_detect(feat,model):
    input_var = feat[np.newaxis,np.newaxis, :, :].astype(np.float32)
    input_var = torch.from_numpy(input_var)
    with torch.no_grad():
        input_var = input_var.cuda()
        preds = model(input_var)
        preds = preds.cpu().detach().numpy()
    return preds


def test_dir(torch_model=None):
    # ========================= initial model ============================
    lstm_flag=0
    multi=False
    l_id=0

    kws_net_file = "/mnt/huanyuan/model/kws_model/asr_english/english_kws_0201.prototxt"
    kws_caffe_model = "/mnt/huanyuan/model/kws_model/asr_english/english_kws_0201.caffemodel"
    kws_net = caffe.Net(kws_net_file,kws_caffe_model,caffe.TEST)
    kws_transformer = caffe.io.Transformer({'data': kws_net.blobs['data'].data.shape})
    kws_transformer.set_transpose('data', (2,0,1))

    data_dir="/home/huanyuan/share/audio_data/english_wav/"
    wav_list=os.listdir(data_dir)
    wav_list.sort()
    # ========================= start test ============================
    kws_count=np.zeros(20)
    f=open('kws_text.txt','w+')
    kws_num=0
    pos=0.0
    sum=0.0
    for id in range(len(wav_list)-1):
        print(id,wav_list[id])
        wav_name = data_dir + wav_list[id]
        if (not os.path.exists(wav_name)):
            continue
        #wavsignal, fs = sf.read('./test_wav/wav/9-0127-asr_16k.wav')
        wavsignal, fs = sf.read(wav_name, dtype='int16')
        if (fs != 16000):
            return [-1], [-1]
        # wavsignal=wavsignal
        wavsignal=wavsignal[0: 48000]
        data_input = GetMfscFeature(wavsignal, fs)
        if(0):
            data_input = data_input * 255 / 10
            data_input = data_input.astype(int)
            a = np.where(data_input > 255)
            data_input[a] = 255
            a = np.where(data_input < 0)
            data_input[a] = 0
            data_input = data_input.astype(np.uint8)
        kws_listall_bp=kws_listall.copy()
        for i in range(0,data_input.shape[0]-295,200):
            data_input_temp=data_input[i:i+296,:]
            if(torch_model):
                out=torch_detect(data_input_temp, torch_model)
            else:
                out=kws_detect(data_input_temp, kws_net, kws_transformer)
            preds = greedy_decode(out)
            for b in range(len(preds)):
                pred_str = ''
                for p in range(len(preds[b])):
                    if ('_' in dicts_bpe[preds[b][p]]):
                        pred_str += ' ' + dicts_bpe[preds[b][p]][1:]
                    elif ('<' in dicts_bpe[preds[b][p]]):
                        pred_str += ' ' + dicts_bpe[preds[b][p]] + ' '
                    else:
                        pred_str += dicts_bpe[preds[b][p]]
                #print('pred: ', pred_str)
                kws_lst=MatchKWS(pred_str)
                if(len(kws_lst) and kws_lst[0] in kws_listall_bp):
                    index=kws_listall_bp.index(kws_lst[0])
                    kws_listall_bp.pop(index)
                #print('kws',kws_lst)
        pos+=len(kws_listall)-len(kws_listall_bp)
        sum+=len(kws_listall)
        print('kws', pos/sum,pos,sum)

if __name__ == '__main__':
    # print('This is main... ')
    # with open('./dataloader/data/english_bpe.txt','r') as dict_file:
    #     dicts_bpe=dict_file.read().splitlines()

    # configfile = open('./config/resnet.yaml')
    # config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    # model = buildmodel(config.model)
    # model=model.cuda()
    # checkpoint = torch.load(config.training.load_model)
    # model.load_state_dict(checkpoint['state_dict'],strict=False)
    # model.eval()

    test_dir()
