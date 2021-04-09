import librosa
import os 

def main():
    input_wav = "/mnt/huanyuan/model/kws_xiaoan8k_test_lmdb/training_audio/xiaoanxiaoan_8k/RM_KWS_XIAOAN_xiaoan_S021M1D30T17.wav"
    input_background_noise_wav = "/mnt/huanyuan/data/speech/kws/xiaoyu_dataset/experimental_dataset/XiaoYuDataset/_background_noise_/white_noise.wav"
    output_dir = "/home/huanyuan/temp"

    sample_rate = 8000
    background_volume_clipped = 0.03

    audio_data = librosa.core.load(input_wav, sr=sample_rate)[0]
    noise_data = librosa.core.load(input_background_noise_wav, sr=sample_rate)[0]
    noise_data = background_volume_clipped * noise_data[:len(audio_data)] 

    print("max: ", audio_data.max(), ", min: ", audio_data.min(), ", abs", min(audio_data.max(), abs(audio_data.min())))
    print("max: ", noise_data.max(), ", min: ", noise_data.min(), ", abs", min(noise_data.max(), abs(noise_data.min())))

    audio_data = audio_data + noise_data

    output_path = os.path.join(output_dir, os.path.basename(input_wav).split('.')[0] + "_add_noise_{}.wav".format(background_volume_clipped))
    librosa.output.write_wav(output_path, audio_data, sr=sample_rate)

if __name__ == "__main__":
    main()