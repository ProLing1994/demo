import librosa
import numpy as np
import pyaudio
import wave

if __name__ == "__main__":
    input_wav = "/home/huanyuan/temp/RM_KWS_XIAORUI_xiaorui_S001M1D00T001_1004.wav"
    # input_wav = "/home/huanyuan/temp/RM_KWS_XIAORUI_xiaorui_S001M1D00T001_1010.wav"
    # input_wav = "/home/huanyuan/temp/RM_KWS_XIAORUI_xiaorui_S001M1D00T001_1016.wav"
    audio_data = librosa.core.load(input_wav, sr=16000)[0]
    print(audio_data)
    # wf = wave.open(input_wav, 'rb')
    
    # idx = 0
    # while True:
    #     data = wf.readframes(16000)
    #     data_np_0 = np.frombuffer(data, dtype = np.int16)
    #     data_np_1 = np.frombuffer(data, np.int16).astype(np.float32) / 32768

    #     pyaudio_listen = pyaudio.PyAudio()
    #     stream = pyaudio_listen.open(format=pyaudio.paInt16,
    #                             channels=1,
    #                             rate=16000,
    #                             output=True)
    #     stream.write(data)
    #     print(idx)
    #     idx += 1