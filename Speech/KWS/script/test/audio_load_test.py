import librosa
import numpy as np
import pyaudio
import wave

def wave_test(input_wav):
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


def librosa_test(input_wav, output_wav):
    audio_data = librosa.core.load(input_wav, sr=8000)[0]
    print(audio_data)
    print("max: ", audio_data.max(), ", min: ", audio_data.min(), ", abs", min(audio_data.max(), abs(audio_data.min())))
    # librosa.output.write_wav(output_wav, audio_data, sr=8000) 

if __name__ == "__main__":
    input_wav = "/mnt/huanyuan/model/kws_xiaoan8k_test_lmdb/training_audio/xiaoanxiaoan_8k/RM_KWS_XIAOAN_xiaoan_S021M1D30T17.wav"
    # input_wav = "/mnt/huanyuan/model/kws_xiaoan8k_test_lmdb/training_audio/xiaoanxiaoan_16k/RM_KWS_XIAOAN_xiaoan_S041M1D01T31.wav"
    # input_wav = "/mnt/huanyuan/model/kws_xiaoan8k_test_lmdb/training_audio/xiaoanxiaoan_16k/RM_KWS_XIAOAN_xiaoan_S021M1D00T21.wav"
    # input_wav = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset/experimental_dataset/XiaoYuDataset/_background_noise_/white_noise.wav"
    output_wav = "/mnt/huanyuan/model/kws_xiaoan8k_test_lmdb/training_audio/xiaoanxiaoan_16k_test/test.wav"

    librosa_test(input_wav, output_wav)
    # wave_test(input_wav)
