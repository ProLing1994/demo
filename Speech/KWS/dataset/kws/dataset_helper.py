import glob
import librosa
import numpy as np
import os
import pcen
import pickle
import random
import sys
import torch

SILENCE_LABEL = '_silence_'
UNKNOWN_WORD_LABEL = '_unknown_'
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech/KWS/script/dataset_align')
sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS/script/dataset_align')
from src.utils.file_tool import read_file_gen

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from ASR.impl.asr_feature_pyimpl import Feature

def load_label_index(positive_label, negative_label):
    # data index
    label_index = {}
    index = 0 
    for _, negative_word in enumerate(negative_label):
        label_index[negative_word] = index
        index += 1
    for _, positive_word in enumerate(positive_label):
        label_index[positive_word] = index
        index += 1
    return label_index


def load_preload_audio(audio_file, audio_idx, audio_label, input_dir, refilename=True):
    # load data
    if refilename:
        if audio_label == SILENCE_LABEL:
            filename = audio_label + '_' + str(audio_idx) + '.txt'
        else:
            filename = os.path.basename(os.path.dirname(audio_file)) + '_' + os.path.basename(audio_file).split('.')[0] + '.txt'
    else:
        filename =  os.path.basename(audio_file).split('.')[0] + '.txt'

    file_path_list = glob.glob(os.path.join(input_dir, '*' + filename).encode('utf-8'))
    assert len(file_path_list) == 1, "{} {}".format(len(file_path_list), os.path.join(input_dir, '*' + filename).encode('utf-8'))
    f = open(file_path_list[0], 'rb')
    data = pickle.load(f)
    f.close()
    return data, filename


def add_frequence_mask(spec, F=27, num_masks=1, replace_with_zero=False, static=False):
    num_mel_channels = spec.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)

        if static:
            f = F

        f_start = random.randrange(0, num_mel_channels - f)
        f_end = f_start + f

        # avoids randrange error if values are equal and range is empty
        if (f_start == f_start + f):
            return spec

        if (replace_with_zero):
            spec[:, f_start:f_end] = 0
        else:
            spec[:, f_start:f_end] = spec.mean()

    return spec


def add_time_mask(spec, T=40, num_masks=1, replace_with_zero=False, static=False):
    len_spectro = spec.shape[0]
    
    for i in range(0, num_masks):
        t = random.randrange(0, T)

        if static:
            t = T

        t_start = random.randrange(0, len_spectro - t)
        t_end = t_start + t

        # avoids randrange error if values are equal and range is empty
        if (t_start == t_start + t):
            return spec

        if (replace_with_zero): 
            spec[t_start:t_end] = 0
        else: 
            spec[t_start:t_end] = spec.mean()

    return spec

class AudioPreprocessor(object):
    def __init__(self, sr=16000, n_mels=40, nfilt=40, n_dct_filters=40, f_max=4000, f_min=20, winlen=0.032, winstep=0.010, data_length=2):
        super().__init__()
        self.sr = sr
        self.n_mels = n_mels
        self.nfilt = nfilt
        self.n_dct_filters = n_dct_filters
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min        
        self.winlen = winlen
        self.winstep = winstep
        self.win_length = int(self.sr * self.winlen)
        self.hop_length = int(self.sr * self.winstep)
        self.data_length = data_length

        self.dct_filters = librosa.filters.dct(self.n_dct_filters, self.n_mels)
        self.pcen_transform = pcen.StreamingPCENTransform(n_mels=self.n_mels, n_fft=self.win_length, hop_length=self.hop_length, trainable=True)

    def compute_mfccs(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.win_length,
            fmin=self.f_min,
            fmax=self.f_max)
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(self.dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").astype(np.float32)
        return data

    def compute_fbanks(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.win_length,
            fmin=self.f_min,
            fmax=self.f_max)
        data[data > 0] = np.log(data[data > 0])
        data = data.T
        return data

    def compute_pcen(self, data):
        data = torch.from_numpy(np.expand_dims(data, axis=0))
        data = self.pcen_transform(data)
        self.pcen_transform.reset()
        data = data.detach().numpy()
        data = data.reshape(-1, 40)
        return data

    def compute_fbanks_cpu(self, data):
        # data to numpy
        data = data * pow(2,15)
        data = data.astype(int)
        # print(data[:10])
        
        # compute fbank cpu
        featurefbanks_cpu = Feature(sample_rate=self.sr, data_length=self.data_length, feature_freq=self.n_mels, nfilt=self.nfilt, winlen=self.winlen , winstep=self.winstep)
        featurefbanks_cpu.get_mel_int_feature(data, len(data))
        feature_data = featurefbanks_cpu.copy_mfsc_feature_int_to()
        return feature_data


# data align
def load_label_align_index(positive_label, positive_label_chinese_name_list, negative_label, align_type='word'):
    # data index
    label_index = {}
    label_align_index = {} 
    index = 0 
    for _, negative_word in enumerate(negative_label):
        label_index[negative_word] = index
        index += 1
    for _, positive_word in enumerate(positive_label):
        label_index[positive_word] = index
        index += 1

    for _, negative_word in enumerate(negative_label):
        label_align_index[negative_word] = 0

    if align_type == "transform":
        for label_idx in range(len(positive_label)):
            positive_label_chinese_name = positive_label_chinese_name_list[label_idx]
            keyword_list = positive_label_chinese_name.split(',')

            if len(keyword_list) < 4:
                continue

            label_list = ["".join([keyword_list[0], keyword_list[1]]),
                        "".join([keyword_list[1], keyword_list[2]]),
                        "".join([keyword_list[2], keyword_list[3]]),]

            for idx, keyword in enumerate(label_list):
                label_align_index[keyword] = idx % 2 + 1

    elif align_type == "word":
        for label_idx in range(len(positive_label)):
            positive_label_chinese_name = positive_label_chinese_name_list[label_idx]
            keyword_list = positive_label_chinese_name.split(',')

            if len(keyword_list) < 4:
                continue

            label_list = [keyword_list[0], keyword_list[1], keyword_list[2], keyword_list[3]]

            for idx, keyword in enumerate(label_list):
                label_align_index[keyword] = idx % 2 + 1

    else:
        raise Exception("[ERROR] Unknow align_type: {}, please check!".fomrat(align_type))

    return label_index, label_align_index


def read_utt2wav(wavscps):
    utt2wav = {}
    for wavscp in wavscps:
        curr_utt2wav = dict({line.split()[0]:line.split()[1] for line in open(wavscp, encoding="utf-8")})
        # merge dict
        utt2wav = {**utt2wav, **curr_utt2wav}
    print("utt2wav:", len(list(utt2wav)))
    return utt2wav


def read_wav2utt(wavscps):
    wav2utt = {}
    for wavscp in wavscps:
        curr_wav2utt = dict({line.split()[1]:line.split()[0] for line in open(wavscp, encoding="utf-8")})
        # merge dict
        wav2utt = {**wav2utt, **curr_wav2utt}
    print("wav2utt:", len(list(wav2utt)))
    return wav2utt


def get_words_dict(ctm_file, keyword_list, align_type='word'):
    content_dict = {}
    word_segments = {}
    
    for index, items in read_file_gen(ctm_file):
        if items[0] not in content_dict.keys():
           content_dict[items[0]] = {}
        if items[4] in content_dict[items[0]].keys():
            content_dict[items[0]][items[4] + "#"] = items
        else:
            content_dict[items[0]][items[4]] = items

    for utt_id in content_dict.keys():
        content = content_dict[utt_id]
        try: 
            word_segments[utt_id] = []

            if align_type == "transform":
                word_segments[utt_id].append([keyword_list[0] + keyword_list[1], 
                                            float(content[keyword_list[0]][2]) + float(content[keyword_list[0]][3])])
                word_segments[utt_id].append([keyword_list[1] + keyword_list[2], 
                                            float(content[keyword_list[1]][2]) + float(content[keyword_list[1]][3])])
                word_segments[utt_id].append([keyword_list[2] + keyword_list[3], 
                                            float(content[keyword_list[2]][2]) + float(content[keyword_list[2]][3])])
            elif align_type == "word":
                word_segments[utt_id].append([keyword_list[0], 
                                            float(content[keyword_list[0]][2]) + float(content[keyword_list[0]][3]) / 2.0])
                word_segments[utt_id].append([keyword_list[1], 
                                            float(content[keyword_list[1]][2]) + float(content[keyword_list[1]][3]) / 2.0])
                word_segments[utt_id].append([keyword_list[2],  
                                            float(content[keyword_list[2]][2]) + float(content[keyword_list[2]][3]) / 2.0])
                word_segments[utt_id].append([keyword_list[3], 
                                            float(content[keyword_list[3]][2]) + float(content[keyword_list[3]][3]) / 2.0])
            else:
                raise Exception("[ERROR] Unknow align_type: {}, please check!".fomrat(align_type))
        except:
            print(utt_id)
    return word_segments


def extract_words(ctm_file, keyword_list, align_type='word'):
    word_segments = get_words_dict(ctm_file, keyword_list, align_type)
    print("word_segments:", len(word_segments.keys()))
    return word_segments