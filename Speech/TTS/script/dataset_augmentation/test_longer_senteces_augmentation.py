import librosa
import numpy as np
import os 
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio


def gen_spacing(sr, sec) : 
    return np.zeros((int(sr * sec), ), dtype=np.float32)


def main():
    input_wav = "/mnt/huanyuan/data/speech/asr/Chinese/BZNSYP/Wave/000001.wav"
    output_dir = "/home/huanyuan/temp"
    audio_data = librosa.core.load(input_wav, sr=16000)[0]
    
    # #3, 0.15
    # spacing = gen_spacing(16000, 0.15)
    # #4, 0.20
    spacing = gen_spacing(16000, 0.20)
    audio_data = np.concatenate((audio_data, spacing, audio_data), axis=0)
    
    output_path = os.path.join(output_dir, os.path.basename(input_wav).split('.')[0] + "_longer_senteces.wav")
    audio.save_wav(audio_data.copy(), output_path, 16000)

if __name__ == "__main__":
    main()