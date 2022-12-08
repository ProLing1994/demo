import librosa
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio

if __name__ == '__main__':

    input_wav = "/home/huanyuan/temp/6maike.wav"
    output_wav = "/home/huanyuan/temp/6maike_vad.wav"
    sampling_rate = 16000

    # load wave
    wav, source_sr = librosa.load(str(input_wav), sr=sampling_rate)
    wav, wav_mask = audio.trim_long_silences(wav, sampling_rate, return_mask_bool=True)
    audio.save_wav(wav, output_wav,sampling_rate)