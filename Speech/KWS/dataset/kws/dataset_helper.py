import sys

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech')
sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from KWS.script.dataset_align.src.utils.file_tool import read_file_gen


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