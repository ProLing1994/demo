import librosa
import numpy as np
from pathlib import Path
import webrtcvad
from scipy.ndimage.morphology import binary_dilation
import struct
import soundfile as sf
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio


if __name__ == '__main__':

    input_wav = "/mnt/huanyuan/data/speech/asr/Chinese/SLR68/train/5_541/5_541_20170607142131.wav"
    output_wav = '/home/huanyuan/temp/vad.wav'
    sampling_rate = 16000

    wav = audio.preprocess_wav(input_wav, sampling_rate)
    sf.write(output_wav, wav, sampling_rate)