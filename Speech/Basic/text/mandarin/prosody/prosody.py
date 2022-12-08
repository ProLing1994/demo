import re


def parse_cn_prosody_label_type1(text, pinyin):
    """
    Parse label from text and pronunciation lines with prosodic structure labelings

    Input text:    妈妈#1当时#1表示#3，儿子#1开心得#2像花儿#1一样#4。
    Input pinyin:  ma1 ma1 dang1 shi2 biao3 shi4 er2 zi5 kai1 xin1 de5 xiang4 huar1 yi2 yang4
    Return pinyin: ma1-ma1 dang1-shi2 biao3-shi4, er2-zi5 kai1-xin1-de5 / xiang4-huar1 yi2-yang4.

    Args:
        - text: Chinese characters with prosodic structure labeling, begin with sentence id for wav and label file
        - pinyin: Pinyin pronunciations, with tone 1-5

    Returns:
        - pinyin&tag: contains pinyin string with optional prosodic structure tags
    """
    text = text.strip()
    pinyin = pinyin.strip()

    # remove punctuations
    text = re.sub('[“”、，。：；？！—…#（）]', '', text)

    # split into sub-terms
    texts  = text
    phones = pinyin.split()

    # prosody boundary tag (SYL: 音节, PWD: 韵律词, PPH: 韵律短语, IPH: 语调短语, SEN: 语句)
    SYL = '-'
    PWD = ' '
    PPH = ' / '
    IPH = ', '
    SEN = '. '

    # parse details
    pinyin = ''
    i = 0 # texts index
    j = 0 # phones index
    b = 1 # left is boundary
    while i < len(texts):
        if texts[i].isdigit():
            if texts[i] == '1': pinyin += PWD  # Prosodic Word, 韵律词边界
            if texts[i] == '2': pinyin += PPH  # Prosodic Phrase, 韵律短语边界
            if texts[i] == '3': pinyin += IPH  # Intonation Phrase, 语调短语边界
            if texts[i] == '4': pinyin += SEN  # Sentence, 语句结束
            b  = 1
            i += 1
        elif texts[i]!='儿' or j==0 or not is_erhua(phones[j-1][:-1]): # Chinese segment
            if b == 0: pinyin += SYL  # Syllable, 音节边界（韵律词内部）
            pinyin += phones[j]
            b  = 0
            i += 1
            j += 1
        else: # 儿化音
            i += 1

    return pinyin


def parse_cn_prosody_label_type2(text, pinyin):
    """
    Parse label from text and pronunciation lines with prosodic structure labelings

    Input text:    /为临帖/他*还|远游|西*安|碑林/龙门|石窟/泰山|*摩崖|石刻/./
    Input pinyin:  wei4 lin2 tie4 ta1 hai2 yuan3 you2 xi1 an1 bei1 lin2 long2 men2 shi2 ku1 tai4 shan1 mo2 ya2 shi2 ke4
    Return pinyin: wei4-lin2-tie4 / ta1-hai2 yuan3-you2 xi1-an1 bei1-lin2 / long2-men2 shi2-ku1 / tai4-shan1 mo2-ya2 shi2-ke4.

    Args:
        - text: Chinese characters with prosodic structure labeling, begin with sentence id for wav and label file
        - pinyin: Pinyin pronunciations, with tone 1-5

    Returns:
        - pinyin&tag: contains pinyin string with optional prosodic structure tags
    """
    # split into sub-terms
    texts = text
    phones = pinyin.strip().split()

    # normalize the text
    texts = re.sub('[ \t\*;?!,.；？！，。]', '', texts)
    texts = texts.replace('//', '/')
    texts = texts.replace('||', '|')
    if texts[0]  in ['/', '|']: texts = texts[1:]
    if texts[-1] in ['/', '|']: texts = texts[:-1]+'.'

    # prosody boundary tag (SYL: 音节, PWD: 韵律词, PPH: 韵律短语, IPH: 语调短语, SEN: 语句)
    SYL = '-'
    PWD = ' '
    PPH = ' / '
    IPH = ', '
    SEN = '.'

    # parse details
    pinyin = ''
    i = 0 # texts index
    j = 0 # phones index
    b = 1 # left is boundary
    while i < len(texts):
        if texts[i] in ['|', '/', ',', '.']:
            if texts[i] == '|': pinyin += PWD  # Prosodic Word, 韵律词边界
            if texts[i] == '/': pinyin += PPH  # Prosodic Phrase, 韵律短语边界
            if texts[i] == ',': pinyin += IPH  # Intonation Phrase, 语调短语边界
            if texts[i] == '.': pinyin += SEN  # Sentence, 语句结束
            b  = 1
            i += 1
        elif texts[i]!='儿' or j==0 or not is_erhua(phones[j-1][:-1]): # Chinese segment
            if b == 0: pinyin += SYL  # Syllable, 音节边界（韵律词内部）
            pinyin += phones[j]
            b  = 0
            i += 1
            j += 1
        else: # 儿化音
            i += 1
    pinyin = pinyin.replace('E', 'ev') # 特殊发音E->ev

    return pinyin


def is_erhua(pinyin):
    """
    Decide whether pinyin (without tone number) is retroflex (Erhua)
    """
    if len(pinyin)<=1 or pinyin == 'er':
        return False
    elif pinyin[-1] == 'r':
        return True
    else:
        return False

