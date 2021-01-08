import argparse
import librosa
import os
import pandas as pd
import re

from tqdm import tqdm

# read file
def read_file_gen(file_path, encoding="utf-8", to_be_split=" "):
    with open(file_path, encoding=encoding) as f:
        all_lines = f.readlines()
        for index, single_line in enumerate(all_lines):
            single_items = single_line.replace("\n", "").split(to_be_split)
            yield index, single_items


def time2second(time):
    h, m, s = time.strip().split(":")
    if "," in s:
        return int(h) * 3600 + int(m) * 60 + int(s.split(',')[0]) + float(s.split(',')[1]) * 0.001
    return int(h) * 3600 + int(m) * 60 + int(s)


def load_srt(args, srt_file):
    srt_list = []       # [{'idx':(), 'start_time':(), 'end_time':(), 'srt':()}]
    srt_dict = {}
    srt_idx = 1
    for item_idx, items in read_file_gen(srt_file, encoding=args.file_encoding):
        if len(items) == 1 and (items[0] == str(srt_idx) or items[0] == '\ufeff1'):
            if srt_dict:
                srt_list.append(srt_dict)
            srt_dict = {} 
            srt_dict['idx'] = srt_idx
            srt_idx += 1
        elif len(items) == 3 and '-->' in items:
            srt_dict['start_time'] = items[0]
            srt_dict['end_time'] = items[2]
        elif len(items) == 1 and items[0] == "":
            continue
        else:
            if 'srt' in srt_dict:
                srt_dict['srt'] += " ".join(items)
            else:
                srt_dict['srt'] = " ".join(items)
    
    # end 
    if srt_dict:
        srt_list.append(srt_dict)
        srt_dict = {}
    return srt_list


def clean_srt_chinese(srt):
    srt = srt.replace(' ', '')
    srt = srt.replace('?', '')
    srt = srt.replace('!', '')
    srt = srt.replace(',', '')
    srt = srt.replace('.', '')
    srt = srt.replace(':', '')
    srt = srt.replace('-', '')
    srt = srt.replace('《', '')
    srt = srt.replace('》', '')
    srt = srt.replace('(', '')
    srt = srt.replace(')', '')

    srt = srt.replace('，', '')
    srt = srt.replace('、', '')
    srt = srt.replace('。', '')
    srt = srt.replace('「', '')
    srt = srt.replace('」', '')
    srt = srt.replace('？', '')
    srt = srt.replace('！', '')
    srt = srt.replace('…', '')
    srt = srt.replace('”', '')
    srt = srt.replace('“', '')
    srt = srt.replace('（', '')
    srt = srt.replace('）', '')

    srt = re.sub(' +', '', srt)

    dict_number = {"0":u"零","1":u"一","2":u"二","3":u"三","4":u"四","5":u"五","6":u"六","7":u"七","8":u"八","9":u"九"}
    for key, value in dict_number.items():
        srt = srt.replace(key, value)
    return srt


def clean_srt_english(srt):
    srt = srt.replace('<i>', ' ')
    srt = srt.replace('</i>', ' ')
    srt = srt.replace('?', ' ')
    srt = srt.replace('!', ' ')
    srt = srt.replace(',', ' ')
    srt = srt.replace('.', ' ')
    srt = srt.replace(':', ' ')
    srt = srt.replace('-', ' ')
    srt = srt.replace('《', ' ')
    srt = srt.replace('》', ' ')
    srt = srt.replace('"', ' ')
    srt = srt.replace('%', ' percent ')
    srt = re.sub(' +', ' ', srt)
    srt = srt.lstrip().rstrip()
    return srt


def merge_srt(args, srt_list):
    srt_clean_list = []         # [{'idx':(), 'start_time':(), 'end_time':(), 'length':(), 'srt':()}]
    srt_failed_list = []         # [{'idx':(), 'start_time':(), 'end_time':(), 'length':(), 'srt':()}]
    srt_dict = {}
    srt_idx = 1
    
    for idx in range(len(srt_list)):
        str_item = srt_list[idx]

        # check 
        if 'srt' not in str_item or 'start_time' not in str_item or 'end_time' not in str_item:
            srt_failed_list.append(str_item)
            continue
        
        srt_dict_temp = {}
        srt_dict_temp['idx'] = srt_idx
        srt_dict_temp['start_time'] = str_item['start_time']
        srt_dict_temp['end_time'] = str_item['end_time']
        srt_dict_temp['length'] = time2second(str_item['end_time']) - time2second(str_item['start_time'])
        srt_dict_temp['srt'] = str_item['srt']

        if args.language == "Chinese":
            if srt_dict_temp['srt'].startswith('(') and srt_dict_temp['srt'].endswith(')'):
                srt_failed_list.append(str_item)
                continue
            if srt_dict_temp['srt'].startswith('「') and srt_dict_temp['srt'].endswith('」'):
                srt_failed_list.append(str_item)
                continue
            if bool(re.search('[a-z]', srt_dict_temp['srt'], re.I)):
                srt_failed_list.append(str_item)
                continue
            srt_dict_temp['srt'] = clean_srt_chinese(srt_dict_temp['srt'])
        elif args.language == "English": 
            srt_dict_temp['srt'] = clean_srt_english(srt_dict_temp['srt'])
        else:
            raise Exception("[ERROR:] Unknow language, please check!")

        if srt_dict:
            if args.language == "Chinese":
                srt = srt_dict['srt'] + srt_dict_temp['srt']
            elif args.language == "English": 
                srt = srt_dict['srt'] + ' ' + srt_dict_temp['srt']
            else:
                raise Exception("[ERROR:] Unknow language, please check!")   

            srt_length = time2second(srt_dict_temp['end_time']) - time2second(srt_dict['start_time'])
            if srt_length > args.max_length_second:

                # check
                # 字幕中，不希望包含剔除的字幕段
                bool_find_failed_srt = False
                for failed_idx in range(len(srt_failed_list)):
                    srt_failed = srt_failed_list[failed_idx]
                    if time2second(srt_dict['start_time']) < time2second(srt_failed['start_time']) \
                        and time2second(srt_dict['end_time']) > time2second(srt_failed['end_time']):
                        bool_find_failed_srt = True
                        break
                
                if not bool_find_failed_srt:
                    srt_clean_list.append(srt_dict)
                    srt_idx += 1

                srt_dict = {}
                srt_dict = srt_dict_temp
                srt_dict['idx'] = srt_idx
            else:
                srt_dict['end_time'] = srt_dict_temp['end_time']
                srt_dict['length'] = srt_length
                srt_dict['srt'] = srt
                if srt_length > args.min_length_second:
                    srt_clean_list.append(srt_dict)
                    srt_dict = {}
                    srt_idx += 1

        else:
            if srt_dict_temp['length'] > args.max_length_second:
                print("[Warring:] 字幕: {}, 长度超过预测最大长度：{}s/{}s".format(srt_dict_temp['srt'], srt_dict_temp['length'], args.max_length_second))
            elif srt_dict_temp['length'] > args.min_length_second:
                srt_dict = srt_dict_temp
                srt_clean_list.append(srt_dict)
                srt_dict = {}
                srt_idx += 1
            else:
                srt_dict = srt_dict_temp

    # end
    if srt_length <= args.max_length_second:
        if srt_dict:
            srt_clean_list.append(srt_dict)
            srt_dict = {}
            srt_idx += 1
            srt_dict = srt_dict_temp
            srt_dict['idx'] = srt_idx
    return srt_clean_list


def audio_split_subtitle(args):
    audio_path = args.audio_path
    assert audio_path.endswith('.wav'), "[ERROR:] Only support wav data"

    output_dir = os.path.join(args.output_dir, os.path.basename(audio_path).split('.')[0])
    # assert not os.path.exists(output_dir), "[ERROR:] Please remove directory: {}, firstly".format(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # init
    sample_rate = 16000
    time_shift_symbol = args.time_shift.split(',')[0]
    time_shift_second = float(args.time_shift.split(',')[1])

    # load srt
    print("Load srt: ")
    srt_list = load_srt(args, args.subtitle_path)
    print("Load srt Done!")

    # merge srt
    print("Merge srt: ")
    srt_list = merge_srt(args, srt_list)
    print("Merge srt Done!")

    # load audio
    print("Load audio: ")
    audio_data = librosa.core.load(args.audio_path, sr=sample_rate)[0]
    print("Load audio Done!")

    # audio split
    print("Split audio: ")
    for srt_idx in tqdm(range(len(srt_list))):
        srt = srt_list[srt_idx]
        start_time = max(0, time2second(srt['start_time']) * 1000 - 200)        # - 200 ms
        end_time = min(len(audio_data) * 1000 / sample_rate, time2second(srt['end_time']) * 1000 + 200)             # + 200 ms

        start_samples = int(sample_rate * start_time / 1000)
        end_samples = int(sample_rate * end_time / 1000)

        audio_sample = audio_data[start_samples: end_samples]

        # out 
        output_path = os.path.join(output_dir, args.output_format.format(args.movie_id, srt['idx']))
        librosa.output.write_wav(output_path, audio_sample, sample_rate) 
    print("Split audio Done!")

    # output srt 
    srt_pd = pd.DataFrame(srt_list)
    srt_pd.to_csv(os.path.join(output_dir, 'srt.csv'), index=False, encoding="utf_8_sig")

def main():
    parser = argparse.ArgumentParser(description="Audio Split Using Subtitle")
    parser.add_argument('--audio_path', type=str, default="E:\\迅雷下载\\mkv\\六福喜事\\六福喜事.wav") 
    parser.add_argument('--subtitle_path', type=str, default="E:\\迅雷下载\\mkv\\六福喜事\\六福喜事.srt") 
    parser.add_argument('--output_dir', type=str, default="E:\\迅雷下载\\mkv\\六福喜事\\")
    parser.add_argument('--language', type=str, choices=["Chinese", "English"], default="Chinese")
    parser.add_argument('--file_encoding', type=str, choices=["gbk", "utf-8"], default="gbk")
    parser.add_argument('--time_shift', type=str, default="+,0.0")
    parser.add_argument('--output_format', type=str, default="RM_MOVIE_S{:0>3d}T{:0>3d}.wav")
    parser.add_argument('--movie_id', type=int, default=1)
    parser.add_argument('--min_length_second', type=int, default=8)
    parser.add_argument('--max_length_second', type=int, default=10)
    args = parser.parse_args()

    audio_split_subtitle(args)


if __name__ == "__main__":
    main()
