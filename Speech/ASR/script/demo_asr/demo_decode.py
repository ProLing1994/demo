import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/ASR')
from impl.asr_decode_pympl import Decode

if __name__ == "__main__":
    test_string_len = 10
    txt_path = "/home/huanyuan/share/audio_data/english_wav/transcript.txt"
    kws_list=[['start', 'record'], ['stop', 'record'], ['mute', 'audio'], ['unmute', 'audio'], 
                ['shot', 'fire'], ['freeze'], ['drop', 'gun'], ['keep', 'hand'], ['put', 'hand'],['down', 'ground']]

    decode_string = None
    with open(txt_path, "r") as f :
        decode_string = f.readlines()[0]

    decode_string_list = decode_string.split(" ")
    for idx in range(len(decode_string_list)):
        if idx + test_string_len >= len(decode_string_list):
            break

        string_idx = decode_string_list[idx : idx + test_string_len]
        # string_idx = ['start', 'stop', 'record', 'unmute', 'mute', 'audio', 'fire', 'freeze', 'audio', 'shot', 'fire']
        print("[Information:] Input: ", string_idx)

        # decode 
        decode = Decode()
        decode_result = decode.match_kws_english(string_idx, kws_list)
        print("[Information:] Result: ", decode_result)