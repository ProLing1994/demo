import argparse
import multiprocessing
import os
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.dataset import audio

def vad(in_params):
    data_path, output_folder = in_params[0], in_params[1]
    sampling_rate = 16000
    wav = audio.preprocess_wav(data_path, sampling_rate)
    audio.save_wav(os.path.join(output_folder, os.path.basename(data_path)), wav, sampling_rate)


def vad_multiprocessing(args):
    # mkdir 
    os.makedirs(args.output_folder)
    
    data_list = os.listdir(args.data_path)

    in_params = []
    for idx in range(len(data_list)):
        data_path = os.path.join(args.data_path, data_list[idx])
        if data_path.endswith(".wav"):
            in_params.append([data_path, args.output_folder])

    p = multiprocessing.Pool( 4 )
    out = list(tqdm(p.imap(vad, in_params), "vad", total=len(data_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # args.data_path = "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/RM_Meiguo_BwcKeyword/danbing_16k/wav_check/"
    # args.output_folder = "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/RM_Meiguo_BwcKeyword/danbing_16k/wav_check_vad/"
    args.data_path = "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/LibriSpeech/train-other-500_check/activatebwc/"
    args.output_folder = "/mnt/huanyuan/data/speech/Recording/RM_Meiguo_Activatebwc/tts/sv2tts/LibriSpeech/train-other-500_check_vad/activatebwc/"
    vad_multiprocessing(args)
