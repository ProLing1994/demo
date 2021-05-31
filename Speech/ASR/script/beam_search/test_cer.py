import os
import sys
import argparse
import time
sys.path.append('/data/engineer/lbj/ASR/torch_version')

import cv2
import torch
import torch.nn.parallel
import matplotlib.pyplot as plt
import torch.optim
import numpy as np
from asr.networks import deepspeech,network_nofc,network_lianyong,ASR_english_phoneme
from asr.general_function.get_mfcc import GetMfscFeature
from asr.rm_asr.simple_beam_search import ctc_beam_search,Ken_LM
import scipy.io.wavfile as wav
import configparser
import wave
import random
import soundfile as sf


algorithem_dict={'0':'Mandarin_Normal','1':'Mandarin_Taxi','2':'Mandarin_Railway','3':'Mandarin_Buslounge','4':'English_BWC_Phoneme'}

label_gt=['xian', 'zai', 'kai', 'shi', 'ce','shi', 'guan', 'bi', 'che', 'chuang', 'guan', 'bi',
          'kong', 'tiao', 'kai', 'che', 'zai', 'lu', 'shang ',
          'ta', 'ma', 'de', 'wo', 'ri', 'ma', 'le', 'ge', 'bi',
       'nao', 'can', 'qu', 'bu', 'liao', 'dai', 'bu', 'liao', 'ni', 'zou', 'bu', 'liao', 'bu', 'qu', 'bu', 'qu', 'bu', 'shun', 'lu',
       'bu', 'zhi', 'dao', 'lu', 'yao', 'jia', 'qian', 'zai', 'duo', 'jia', 'dian', 'da', 'biao', 'jiu', 'bu', 'qu',
       'da','biao', 'bu', 'xing', 'nong', 'si', 'ni', 'yao', 'shi', 'gan', 'bao', 'jing', 'shou', 'ji', 'na', 'chu', 'lai',
       'zi', 'ji', 'jie', 'suo', 'hai', 'shi', 'wo', 'bang', 'ni', 'yao', 'qian', 'hai', 'shi', 'yao', 'ming',
       'ba', 'qian', 'na', 'chu', 'lai', 'bie', 'luan', 'dong', 'xiao', 'rui', 'xiao', 'rui', 'ni', 'hao', 'xiao', 'rui',
       'da', 'kai', 'di', 'tu', 'da', 'kai', 'lan', 'ya', 'da', 'kai', 'shou', 'yin', 'ji', 'da', 'dian', 'hua', 'cha', 'kan', 'ying', 'shou',
       'qiang', 'dan', 'yin', 'liang', 'jia', 'da', 'yin', 'liang', 'jian', 'xiao', 'ce', 'shi', 'wan', 'bi']

kws2=[['ta','ma','de'],['wo','ri'],['ma','le','ge','bi'],['nao','can'],['qu','bu','liao'],['dai','bu','liao','ni',],
     ['zou','bu','liao',],['bu','qu','bu','qu'],['bu','shun','lu'],['bu','zhi','dao','lu'],['yao','jia','qian'],
     ['zai','duo','jia','dian'],['da','biao','jiu','bu','qu'],['da','biao','bu','xing'],['nong','si','ni'],
     ['yao','shi','gan','bao','jing'],['shou','ji','na','chu','lai'],['zi','ji','jie','suo','hai','shi','wo','bang','ni'],
     ['yao','qian','hai','shi','yao','ming'],['ba','qian','na','chu','lai'],['bie','luan','dong'],
     ['xiao','rui','xiao','rui'],['ni','hao','xiao','rui'],['da','kai','di','tu'],['da','kai','lan','ya'],
     ['da','kai','shou','yin','ji'],['da','dian','hua'],['cha','kan','ying','shou'],['qiang','dan'],
     ['yin','liang','jia','da'],['yin','liang','jian','xiao'],['dao','hang','dao','gao','tie','zhan'],
     ['dao','hang','dao','ke','ji','yuan'],['dao','hang','dao','ji','chang']]

kws=[['ta','ma','de'],['wo','ri'],['wo','cao'],['ma','le','ge','bi'],['nao','can'],['qu','bu','liao'],['dai','bu','liao','ni',],
     ['zou','bu','liao',],['bu','qu','bu','qu'],['bu','shun','lu'],['yao','jia','qian'],
     ['da','biao','jiu','bu','qu'],['da','biao','bu','xing'],['nong','si','ni'],
     ['yao','shi','gan','bao','jing'],['shou','ji','na','chu','lai'],['zi','ji','jie','suo','hai','shi','wo','bang','ni'],
     ['yao','qian','hai','shi','yao','ming'],['ba','qian','na','chu','lai'],
     ['xiao','rui','xiao','rui'],['ni','hao','xiao','rui'],['da','kai','di','tu'],['da','kai','lan','ya'],
     ['da','kai','shou','yin','ji'],['da','dian','hua'],['cha','kan','ying','shou'],['qiang','dan'],['jie','dan'],['zan','ting'],['ji','xu'],
     ['yin','liang','jia','da'],['yin','liang','jian','xiao'],['shang','yi','pin','dao'],['xia','yi','pin','dao'],['dao','hang','dao','shen','zhen','bei','zhan'],
     ['dao','hang','dao','zu','zi','lin','di','tie','zhan'],['dao','hang','dao','ji','chang'],['dao','hang','dao','si','jie','zhi','chuang']]

ignore_tongue_table={}
ignore_tongue_table['li']='ni'
ignore_tongue_table['sou']='shou'
ignore_tongue_table['ling']='lin'
ignore_tongue_table['jin']='jing'
ignore_tongue_table['la']='na'
ignore_tongue_table['lao']='nao'
ignore_tongue_table['chan']='can'
ignore_tongue_table['nan']='lan'



def save_wav(filename,data):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(data.tostring())
    wf.close()



def GetEditDistance_RM(str1,str2):
    '''
    :param str1:input sequence
    :param str2: key words
    :return: distence between input and keywords
    '''
    m=len(str1)
    n=len(str2)
    v=np.zeros([m+1,n+1])
    for i in range(m):
        v[i+1,0]=i+1
    for j in range(n):
        v[0,j+1]=j+1
    for i in range(m):
        for j in range(n):
            if(str1[i]==str2[j]):
                v[i+1,j+1]=v[i,j]
            else:
                v[i+1,j+1]=1+min(min(v[i+1,j],v[i,j+1]),v[i,j])
    leven_cost = v[m,n]

    return leven_cost

def match_kws(input):
    global kws_list
    global pinyin_list
    if(input==''):
        return ''
    kws_tmp=kws_list.copy()
    pinyin_tmp=pinyin_list.copy()
    i=0;
    match_string=''
    #input=input.split()
    slen=len(input)
    if(isinstance(pinyin_list[0][0], list)):
        while (i < slen):
            for j in range(len(pinyin_tmp)):
                a=input[i]
                if (input[i] == pinyin_tmp[j][0][0]):
                    dist = GetEditDistance_RM(input[i:i + len(pinyin_tmp[j][0])], pinyin_tmp[j][0])
                    if (dist == 0):
                        i = i + len(pinyin_tmp[j][0])
                        if(len(pinyin_tmp[j][1])):
                            while(i<slen and '_' not in input[i]):
                                i=i+1
                            if(i==slen):
                                break
                            dist = GetEditDistance_RM(input[i:min(slen,i + len(pinyin_tmp[j][1]))], pinyin_tmp[j][1])
                            if(dist<=1):
                                match_string += kws_tmp[j] + '    '
                                del kws_tmp[j]
                                del pinyin_tmp[j]
                                break
                        else:
                            i=i-1
                            match_string += kws_tmp[j] + '    '
                            del kws_tmp[j]
                            del pinyin_tmp[j]
                            break
            i += 1
    else:
        while(i<slen):
            for j in range(len(pinyin_tmp)):
                if(input[i]==pinyin_tmp[j][0]):
                    dist=GetEditDistance_RM(input[i:i+len(pinyin_tmp[j])],pinyin_tmp[j])
                    if((dist==0) or(len(pinyin_tmp[j])>6 and dist<2) or(len(pinyin_tmp[j])>9 and dist<3)):
                        i = i + len(pinyin_tmp[j]) - 2
                        match_string+=kws_tmp[j]+'    '
                        del kws_tmp[j]
                        del pinyin_tmp[j]
                        break
            i+=1
    return match_string


def greedy_decode(input,rnnlm=None):
    [b,t,c]=input.shape
    out_batch=[]
    if(rnnlm):
        for i in range(b):
            out=[]
            score = np.zeros([407])
            hidden_state=None
            for j in range(t):
                id=np.argmax(input[i,j]==np.max(input[i,j]))
                if(len(out)!=0):
                  if(out[-1]!=id and id!=0):
                      out.append(id)
                      id=torch.tensor(id).cuda()
                      id=id.unsqueeze(0)
                      id=id.unsqueeze(1)
                      score,hidden_state=rnnlm.forward_onestep(id,hidden_state)
                      score=score.cpu().detach().numpy()[0,:407]

                else:
                  out.append(id)
                  id = torch.tensor(id).cuda()
                  id = id.unsqueeze(0)
                  id = id.unsqueeze(1)
                  score, hidden_state = rnnlm.forward_onestep(id, hidden_state)
                  score = score.cpu().detach().numpy()[0, :407]

                if(j<t-1):
                    input[i, j + 1] = input[i, j + 1] + 0.3*score
            out_batch.append(out[1:])
    else:
        for i in range(b):
            out = []
            for j in range(t):
                id = np.argmax(input[i, j] == np.max(input[i, j]))
                if (len(out) != 0):
                    if (out[-1] != id):
                        out.append(id)
                else:
                    out.append(id)
            out_batch.append(out)
    return out_batch

def eval_testset_cer(model,rnnlm=None):
    model.eval()
    data_path='G:/DATASET/speech_recognition/普通话音频/锐明录制/中文关键词录制0205_单兵多麦/'
    result_dict={}
    sub_dir=['降噪前','降噪后']
    for sub in sub_dir:
        test_dirs=os.listdir(data_path+sub+'/')
        with open(data_path+sub+'/' + 'transcript.txt', 'r') as f:
            label_list = f.read().splitlines()
        label_dict = {}
        for l in label_list:
            l = l.split(' ', 1)
            label_dict[l[0]] = l[1]
        for test_dir in test_dirs[1:]:
            wav_list=os.listdir(data_path+sub+'/'+test_dir+'/')
            dist=0
            sum_char=0
            match_num=0
            sum_kws=0
            for w in wav_list[:-1]:
                if(wav_list.index(w)<0):
                    continue
                wav_path=data_path+sub+'/'+test_dir+'/'+w

                w_id=w.split('P')[-1]
                #w_id=int(w_id.split('.')[0])-1
                w_id = str(int(w_id.split('.')[0]))
                if (not os.path.exists(wav_path)):
                    print('wav not exist!', wav_path)
                    return -1
                fs, wavsignal = wav.read(wav_path)
                wavsignal=wavsignal
                #wavsignal=wavsignal[::2]
                #fs=8000
                feat = GetMfscFeature(wavsignal, fs)
                if (1):
                    feat = feat * 255 / 10
                    feat = feat.astype(int)
                    a = np.where(feat > 255)
                    feat[a] = 255
                    a = np.where(feat < 0)
                    feat[a] = 0
                #plt.imshow(feat.T)
                #plt.show()
                input_var = feat[np.newaxis, :, :, np.newaxis].astype(np.float32)
                input_var = input_var.transpose(0, 3, 1, 2)
                input_var = torch.from_numpy(input_var)
                with torch.no_grad():
                    input_var = torch.autograd.Variable(input_var.cuda())
                    fc_outputs = model(input_var)
                    fc_outputs = fc_outputs.cpu().detach().numpy()
                    fc_outputs = fc_outputs[:, :, :407]
                    if(w_id not in label_dict):
                        continue
                    label_b=label_dict[w_id]
                    pred_b = []
                    if(0):
                        out_batch = greedy_decode(fc_outputs, rnnlm)
                        for i in range(len(out_batch)):
                            pred = []
                            for id in out_batch[i]:
                                if (id != 0):
                                    #pred.append(dicts_tone[id][:-1])
                                    if (dicts[id] in ignore_tongue_table):
                                        pred.append(ignore_tongue_table[dicts[id]])
                                    else:
                                        pred.append(dicts[id])
                            pred_b.append(pred)
                    else:
                        out_batch = ctc_beam_search(fc_outputs[0], rnnlm, 5, 0, dicts, bswt=1.0, lmwt=0.3)
                        pred_b.append(out_batch[0]['words'].split())
                    label_b=label_b.split(' ')
                    for i in range(len(label_b)):
                        label_b[i]=label_b[i]
                    label_b=label_b[:-1]
                    dist += GetEditDistance_RM(pred_b[0], label_b)
                    sum_char+=len(label_b)
                    k_list=match_kws(label_b)
                    sum_kws+=len(k_list)
                    if(len(k_list)):
                        match_num += match_kws(pred_b[0],k_list)
            if test_dir in result_dict:
                tmp_str=result_dict[test_dir]
                tmp_str=tmp_str+' CER:'+"%.3f"%(dist * 1.0 /sum_char)+' TPR:'+"%.3f"%(match_num*1.0/sum_kws)
                result_dict[test_dir]=tmp_str
            else:
                tmp_str=test_dir+' CER:'+"%.3f"%(dist * 1.0 /sum_char)+' TPR:'+"%.3f"%(match_num*1.0/sum_kws)
                result_dict[test_dir]=tmp_str
            #print('{}{} pred CER :{} TPR:{}'.format(sub,test_dir,dist * 1.0 /sum_char ,match_num*1.0/sum_kws))
    print('===========test result===========')
    print('=======降噪前============降噪后=====')
    for item in result_dict.keys():
        print(result_dict[item])


def eval_cer(model,lm,algorithem_id):
    model.eval()
    data_path='G:/DATASET/speech_recognition/轨交语料_普通话/测试音频0415/wav/'
    label_path='G:/DATASET/speech_recognition/轨交语料_普通话/测试音频0415/'
    with open(label_path+'transcript_test.txt','r',encoding='utf-8-sig') as f:
        label_list=f.read().splitlines()
    label_dict={}
    for l in label_list:
        l=l.split('\t')
        label_dict[l[0]]=l[1]
    wav_list=os.listdir(data_path)
    has_transcript=True
    sk=0
    dist=0
    sum_dist=0
    sum_char=0
    match_num=0
    sum_kws=0
    for w in wav_list:
        #print(wav_list.index(w),w)
        if(wav_list.index(w)<0):
            continue
        wav_path=data_path+w
        if (not os.path.exists(wav_path)):
            print('wav not exist!', wav_path)
            return -1
        wavsignal,fs = sf.read(wav_path,dtype='int16')
        #wavsignal, fs = sf.read('./test_wav/buslounge_3.wav', dtype='int16')
        if(0):
            wavsignal=wavsignal[:]
            fs=8000
        feat = GetMfscFeature(wavsignal, fs)
        if (1):
            feat = feat * 255 / 10
            feat = feat.astype(int)
            a = np.where(feat > 255)
            feat[a] = 255
            a = np.where(feat < 0)
            feat[a] = 0
        if(0):
            m,n=feat.shape
            s=int(random.random()*(m-300))
            cv2.imwrite('G:/Hisilicon_tools/HISVP2/ConvertModel/model/image_296_56/pic_'+str(sk)+'.jpg',feat[s:s+296])
            sk+=1
        input_var = feat[np.newaxis, :, :, np.newaxis].astype(np.float32)
        input_var = input_var.transpose(0, 3, 1, 2)
        input_var = torch.from_numpy(input_var)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_var.cuda())
            fc_outputs = model(input_var)
            fc_outputs = fc_outputs.cpu().detach().numpy()
            fc_outputs = fc_outputs[:, :, :407]
            '''
            if(w not in label_dict):
                continue
            label_b=label_dict[w]
            print('label words: ',label_b)
            '''
            pred_b = []
            if (0):
                out_batch = greedy_decode(fc_outputs, rnnlm)
                for i in range(len(out_batch)):
                    pred = []
                    for id in out_batch[i]:
                        if (id != 0):
                            # pred.append(dicts_tone[id][:-1])
                            if (dicts[id] in ignore_tongue_table):
                                pred.append(ignore_tongue_table[dicts[id]])
                            else:
                                pred.append(dicts[id])
                    pred_b.append(pred)
            else:
                out_batch = ctc_beam_search(fc_outputs[0], lm, 5, 0, asr_dicts, bswt=1.0, lmwt=0.3)
                pred=out_batch[0]['words']
                pred_b.append(pred)
                print('predict words: ', pred)
                if(has_transcript):
                    label=label_dict[w]
                    dist=GetEditDistance_RM(pred.split(),label.split())
                    sum_dist += dist
                    sum_char += len(label.split())
                    print('wav {} pred CER :{} dist:{} sum:{}'.format(w,sum_dist * 1.0 /sum_char , sum_dist,sum_char))

                else:
                    match_string=match_kws(pred)
                    print('match_string: ',match_string)
            pred_b.append(pred)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def eval(model,lm,algorithem_id):
    model.eval()
    wav_path = './test_wav/test-kws-xiaorui-asr-mandarin.wav'
    fs, wavsignal = wav.read(wav_path)
    if(0):
        wavsignal=wavsignal[::2]
        fs=8000
    feat = GetMfscFeature(wavsignal, fs)
    if (1):
        feat = feat * 255 / 10
        feat = feat.astype(int)
        a = np.where(feat > 255)
        feat[a] = 255
        a = np.where(feat < 0)
        feat[a] = 0
    input_var = feat[np.newaxis, :, :, np.newaxis].astype(np.float32)
    input_var = input_var.transpose(0, 3, 1, 2)
    input_var = torch.from_numpy(input_var)
    with torch.no_grad():
        input_var = torch.autograd.Variable(input_var.cuda())
        fc_outputs,_ = model(input_var)
        fc_outputs = fc_outputs.cpu().detach().numpy()
        print(np.argmax(fc_outputs,axis=-1))
        fc_outputs = fc_outputs[:, :, :-1]
        pred_b = []
        if (0):
            out_batch = greedy_decode(fc_outputs)
            for i in range(len(out_batch)):
                pred = []
                for id in out_batch[i]:
                    if (id != 0):
                        # pred.append(dicts_tone[id][:-1])
                        if (asr_dicts[id] in ignore_tongue_table):
                            pred.append(ignore_tongue_table[asr_dicts[id]])
                        else:
                            pred.append(asr_dicts[id])
                print('pred: ',pred)
                pred_b.append(pred)
        else:
            out_batch = ctc_beam_search(fc_outputs[0], lm, 5, 0, asr_dicts, bswt=1.0, lmwt=0.3)
            pred=out_batch[0]['words']
            print('predict words: ', pred)
            match_string=match_kws(pred)
            print('match_string: ',match_string)
            pred_b.append(pred)



def tmp(model,lm,algorithem_id):
    model.eval()
    wav_path = './test_wav/录音.wav'
    fs, wavsignal = wav.read(wav_path)
    lmresult=[]
    lstm_state=None
    for i in range(0,len(wavsignal)-9632,6400):
        if(i==0):
            feat_t = GetMfscFeature(wavsignal[0:9632], fs)
            feat_t = feat_t * 255 / 10
            feat_t = feat_t.astype(int)
            a = np.where(feat_t > 255)
            feat_t[a] = 255
            a = np.where(feat_t < 0)
            feat_t[a] = 0
        else:
            feat_tmp = GetMfscFeature(wavsignal[9632-512+i-6400:9632+i], fs)
            feat_tmp = feat_tmp * 255 / 10
            feat_tmp = feat_tmp.astype(int)
            a = np.where(feat_tmp > 255)
            feat_tmp[a] = 255
            a = np.where(feat_tmp < 0)
            feat_tmp[a] = 0
            feat_t=np.concatenate((feat_t[40:],feat_tmp),axis=0)

        input_var = feat_t[np.newaxis, :, :, np.newaxis].astype(np.float32)
        input_var = input_var.transpose(0, 3, 1, 2)
        input_var = torch.from_numpy(input_var)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_var.cuda())
            fc_outputs,lstm_state = model(input_var,lstm_state=lstm_state)
            fc_outputs = fc_outputs.cpu().detach().numpy()
            fc_outputs = fc_outputs[:, :, :-1]
            print(np.argmax(fc_outputs,axis=-1))
            out_batch = ctc_beam_search(fc_outputs[0], lm, 5, 0, asr_dicts, bswt=1.0, lmwt=0.3,results=lmresult)
            lmresult=out_batch
            pred=out_batch[0]['words']
            print('predict words: ', pred)

def initASR(checkpoint_file):
    conf = configparser.ConfigParser()
    conf.read(checkpoint_file,encoding='utf-8-sig')
    algorithem_id=int(conf.get('AsrAlgorithem_ID','algorithem_id').split('\t')[0])
    algorithem_kind=algorithem_dict[str(algorithem_id)]
    kws_list = []
    pinyin_list = []
    if(algorithem_id!=0):
        kws_num = int(conf.get(algorithem_kind, 'cls_num')) + 1
        for i in range(1, kws_num):
            kws = conf.get(algorithem_kind, str(i))
            kws = kws.split('=')
            kws_list.append(kws[0])
            if ('\t' in kws[1]):
                tmp = kws[1].split('\t')
                if (tmp[1] == '0'):
                    pinyin_list.append([tmp[0].split(), ''])
                else:
                    pinyin_list.append([tmp[0].split(), tmp[1].split()])
            else:
                pinyin_list.append(kws[1].split())
    if(algorithem_id==4):
        asr_model = ASR_english_phoneme.ASR_English_Net(136)
    elif(algorithem_id==0):
        #asr_model = network_nofc.__dict__['Res18'](3510, pretrained = False)
        asr_model = deepspeech.DeepSpeech(408,256,7)
    else:
        asr_model = network_nofc.__dict__['Res18'](408, pretrained = False)
    asr_model=asr_model.cuda()
    model_path='./checkpoint'
    if(algorithem_id==1):
        checkpoint_file = os.path.join(model_path,'0520taxi/taxi_16k_64dim.pth')
        lm_file = os.path.join(model_path,'LM/3gram_taxi_408.bin')
        dict_file=os.path.join(model_path,'LM/dict_taxi.txt')
    elif(algorithem_id==2):
        checkpoint_file = os.path.join(model_path, '0429/railway_16k_64dim.pth')
        lm_file = os.path.join(model_path,'0429/3gram_railway_408.bin')
        dict_file=os.path.join(model_path,'0429/dict_railway.txt')
    elif(algorithem_id==3):
        checkpoint_file = os.path.join(model_path, '0511BUS/buslounge_16k_64dim.pth')
        lm_file = os.path.join(model_path,'0511BUS/3gram_buslounge_408.bin')
        dict_file=os.path.join(model_path,'0511BUS/dict_buslounge.txt')
    elif(algorithem_id==4):
        checkpoint_file = os.path.join(model_path, '0513/BWC_phoneme_64dim_6.4.pth')
        lm_file = os.path.join(model_path,'0513/4gram_phoneme_136.bin')
        dict_file=os.path.join(model_path,'0513/phoneme_dict.txt')
    else:
        checkpoint_file = os.path.join(model_path,'0518hanzi/deepspeech408.epoch9.pth')
        lm_file = os.path.join(model_path,'LM/3gram_mandarin_408.bin')
        dict_file=os.path.join(model_path,'0518hanzi/hanzi408.txt')

    checkpoint = torch.load(checkpoint_file)
    asr_model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded asr_model '{}'".format(checkpoint_file))
    lm = Ken_LM(lm_file)
    print("=> loaded LM model  '{}'".format(lm_file))
    with open(dict_file, 'r',encoding='utf-8-sig') as f:
        asr_dicts = f.read().splitlines()
    print("=> loaded dict   '{}'".format(dict_file))
    asr_model.eval()
    return asr_model,lm,asr_dicts,kws_list,pinyin_list,algorithem_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Test')

    checkpoint_file = os.path.join('./checkpoint/configFileASR.cfg')
    asr_model,lm,asr_dicts,kws_list,pinyin_list,algorithem_id=initASR(checkpoint_file)

    eval(asr_model,lm,algorithem_id)
    #eval_cer(asr_model,lm,algorithem_id)
    #eval_testset_cer(model,lm)
