import argparse
import librosa
import multiprocessing as mp
import os
import soundfile as sf


from src.utils.file_tool import read_file_gen
from tqdm import tqdm

def read_utt2wav():
    utt2wav = {}
    for wavscp in wavscps:
        curr_utt2wav = dict({line.split()[0]:line.split()[1] for line in open(wavscp)})
        # merge dict
        utt2wav = {**utt2wav, **curr_utt2wav}
    print("utt2wav:", len(list(utt2wav)))
    return utt2wav

def read_signal(utt_id):
    utt_file_path = utt2wav[utt_id]
    # signal,_ = librosa.load(utt_file_path, 16000)
    signal, sr = sf.read(utt_file_path)
    return signal, sr

def cut_word_and_save(items):
    utt_id = items[0]
    word = items[1]
    tbegin = items[2]
    tend = items[3]

    word_save_dir = save_dir + '/' + word + '/'
    if os.path.exists(word_save_dir + os.path.basename(utt2wav[utt_id])):
        return 0

    sig, sr = read_signal(utt_id)
    #sig = sig[:,0]
    if len(sig.shape) > 1:
        keyword_sample = sig[int(sr*tbegin):int(sr*tend),:]
        sf.write(word_save_dir + os.path.basename(utt2wav[utt_id]), keyword_sample, sr)
    else:
        keyword_sample = sig[int(sr*tbegin):int(sr*tend)]
        sf.write(word_save_dir + os.path.basename(utt2wav[utt_id]), keyword_sample, sr)
    return 1

def get_words_list(ctm_file):
    content_dict = {}
    word_segments = []
    
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
            word_segments.append([utt_id, "".join(keyword_list), float(content[keyword_list[0]][2]), float(content[keyword_list[-1]][2]) + float(content[keyword_list[-1]][3])])
        except:
            print(utt_id)
    return word_segments

def extract_words(ctm_file):
    process_num = 30
    word_segments = get_words_list(ctm_file)
    print("word_segments:", len(word_segments))
    with mp.Pool(process_num) as p:
        frames = list(tqdm(p.imap(cut_word_and_save, word_segments), total=len(word_segments)))
    # for idx in range(len(word_segments)):
    #     cut_word_and_save(word_segments[idx])

if __name__ == "__main__":
    default_ctm_file = "/mnt/huanyuan/data/speech/kws/xiaorui_dataset/original_dataset/XiaoRuiDataset_12022020/kaldi_type/tmp/nnet3_align/ctm"
    default_wav_file = "/mnt/huanyuan/data/speech/kws/xiaorui_dataset/original_dataset/XiaoRuiDataset_12022020/kaldi_type/wav.scp"
    default_keyword_list = "小,锐,小#,锐#"
    default_save_dir = "/mnt/huanyuan/data/speech/kws/xiaorui_dataset/original_dataset/XiaoRuiDataset_12022020/kaldi_cut_keyword"

    parser = argparse.ArgumentParser(description = "Cut keyword wavs and save them")
    parser.add_argument('--ctm_file', type=str,default=default_ctm_file,dest='ctm_file',help='Align ctm file path')
    parser.add_argument('--wav_file',type=str,default=default_wav_file,dest='wav_file',help='Wav path')
    parser.add_argument('--keyword_list',type=str,default=default_keyword_list,dest='keyword_list',help='Keyword list')
    parser.add_argument('--save_dir',type=str,default=default_save_dir,dest='save_dir',help='Save destination path')
    args = parser.parse_args();


    ctm_files = [args.ctm_file] # "/NASdata/zhangchx/kaldi/egs/thchs30/s5/exp/tri4b_dnn_mpe/decode_test_word_it3/ctm"]
    wavscps = [args.wav_file] # "/NASdata/zhangchx/kaldi/egs/thchs30/s5/data/test/wav.scp"]
    keyword_list = args.keyword_list.split(',')
    save_dir = args.save_dir
    beg_context = 0 # 3600
    end_context = 0 # 1200

    print("[Begin] Cut keyword wavs")

    os.system("mkdir -p " + save_dir + " && mkdir -p " + save_dir + "/" + "".join(keyword_list))
    utt2wav = read_utt2wav()

    for ctm_file in ctm_files:
        extract_words(ctm_file)

    print("[Done] Cut keyword wavs")

