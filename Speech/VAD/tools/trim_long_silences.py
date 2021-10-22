import sys 

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio


if __name__ == '__main__':

    # input_wav = "/mnt/huanyuan/data/speech/asr/Chinese/SLR68/train/5_541/5_541_20170607142131.wav"
    # output_wav = '/home/huanyuan/temp/vad.wav'
    # sampling_rate = 16000

    input_wav = "/home/huanyuan/temp/1017S001D01017_38_1.wav"
    output_wav = "/home/huanyuan/temp/1017S001D01017_38_1_vad.wav"
    sampling_rate = 8000

    wav = audio.preprocess_wav(input_wav, sampling_rate)
    audio.save_wav(output_wav, wav, sampling_rate)