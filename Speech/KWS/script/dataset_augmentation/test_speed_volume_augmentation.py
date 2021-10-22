import librosa
import os 
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio

def main():
    input_wav = "/home/huanyuan/temp/RM_KWS_XIAORUI_xiaorui_S001M1D00T001.wav"
    output_dir = "/home/huanyuan/temp"
    audio_data = librosa.core.load(input_wav, sr=16000)[0]
    
    speed = 1.0
    volume = 1.0
    pitch = -5
    
    # speed > 1, 加快速度
    # speed < 1, 放慢速度
    audio_data = librosa.effects.time_stretch(audio_data, speed)

    # 音量大小调节
    audio_data = audio_data * volume

    # 音调调节
    audio_data = librosa.effects.pitch_shift(audio_data, sr=16000, n_steps=pitch)

    output_path = os.path.join(output_dir, os.path.basename(input_wav).split('.')[0] + "_spped_{}_volume_{}_pitch_{}.wav".format(speed, volume, pitch))
    audio.save_wav(audio_data.copy(), output_path, 16000)

if __name__ == "__main__":
    main()