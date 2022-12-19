import re
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.config import hparams
from TTS.script.dataset_align.src.utils.file_tool import read_file_gen


def get_words_set(ctm_file):
    words_set = {}
    
    for _, items in read_file_gen(ctm_file):

        utterance = items[0]
        word = items[4]
        t_begin = items[2]
        t_length = items[3]

        if utterance not in words_set:
            words_set[utterance] = []
            words_set[utterance].append([word, t_begin, t_length])
        else:
            words_set[utterance].append([word, t_begin, t_length])
    return words_set


def get_prosody_label_text(words_set, utterance, text):

    text_prosody = ''
    if utterance not in words_set:
        print(utterance)
        return text_prosody

    words = words_set[utterance]

    # text，去掉无用字符
    text = re.sub('[“”、，。：；？！—…#（）]', '', text)
    text = text.strip()
    text_no_symbol = re.sub('[%$]', '', text)

    SYL = ''
    PWD = '#1'
    PPH = '#2'
    IPH = '#3'
    SEN = '#4'

    words_idx = 0
    for text_idx in range(len(text)):
        if text[text_idx] in ['%', '$']:
            assert text_idx - 1 >= 0
            assert words[words_idx - 1][0] == text_no_symbol[words_idx - 1] or words[words_idx - 1][0] == '<UNK>'
            if float(words[words_idx - 1][2]) < hparams.prosody_threshold['1'][0]:
                text_prosody += SYL
            elif float(words[words_idx - 1][2]) < hparams.prosody_threshold['1'][1]:
                text_prosody += PWD
            elif float(words[words_idx - 1][2]) < hparams.prosody_threshold['2'][1]:
                text_prosody += PPH
            elif float(words[words_idx - 1][2]) < hparams.prosody_threshold['3'][1]:
                text_prosody += IPH
            elif float(words[words_idx - 1][2]) < hparams.prosody_threshold['4'][1]:
                text_prosody += SEN
            else:
                text_prosody += SEN
        elif words[words_idx][0] == text_no_symbol[words_idx]:
            text_prosody += text_no_symbol[words_idx]
            words_idx += 1
        else:
            if words[words_idx][0] == '<UNK>':
                if words_idx + 1 >= len(text_no_symbol):
                    pass
                elif words[words_idx + 1][0] == text_no_symbol[words_idx + 1]:
                    words_idx += 1
                elif words[words_idx + 1][0] == text_no_symbol[words_idx]:
                    if words_idx - 1 >= 0:
                        words[words_idx - 1][2] = str(float(words[words_idx - 1][2]) + float(words[words_idx][2]))
                    del words[words_idx]
                elif words[words_idx + 1][0] == '<UNK>':
                    words_idx += 1
                else:
                    print()
            else:
                print()

    return text_prosody