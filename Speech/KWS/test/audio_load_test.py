import librosa
import numpy as np
import pyaudio
import wave

if __name__ == "__main__":
    # input_wav = "/mnt/huanyuan/data/speech/kws/tf_speech_commands/speech_commands/left/00176480_nohash_0.wav"
    input_wav = "/mnt/huanyuan/model/test_straming_wav/test.wav"
    # audio_data = librosa.core.load(input_wav, sr=16000)[0]
    wf = wave.open(input_wav, 'rb')
    
    idx = 0
    while True:
        data = wf.readframes(16000)
        data_np_0 = np.frombuffer(data, dtype = np.int16)
        data_np_1 = np.frombuffer(data, np.int16).astype(np.float32) / 32768

        pyaudio_listen = pyaudio.PyAudio()
        stream = pyaudio_listen.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=16000,
                                output=True)
        stream.write(data)
        print(idx)
        idx += 1