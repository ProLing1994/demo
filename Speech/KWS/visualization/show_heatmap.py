import argparse
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from impl.pred_pyimpl import kws_load_model
from impl.pred_pyimpl import audio_preprocess
from utils.train_tools import *
 

def load_data(args, cfg, sample_rate, desired_samples):
    audio_data = librosa.core.load(args.input_wav, sr=sample_rate)[0]

    # alignment data
    if len(audio_data) < desired_samples:
        data_length = len(audio_data)
        audio_data = np.pad(audio_data, (max(0, (desired_samples - data_length)//2), 0), "constant")
        audio_data = np.pad(audio_data, (0, max(0, (desired_samples - data_length + 1)//2)), "constant")

    if len(audio_data) > desired_samples:
      data_offset = np.random.randint(0, len(audio_data) - desired_samples)
      audio_data = audio_data[data_offset:(data_offset + desired_samples)]

    assert len(audio_data) == desired_samples, "[ERROR:] Something wronge about audio length, please check"
    
    # audio preprocess, load mfcc data
    audio_data = audio_preprocess(cfg, audio_data)

    # to tensor
    data_tensor = torch.from_numpy(audio_data.reshape(1, -1, 40))
    data_tensor = data_tensor.float()
    return audio_data, data_tensor


def genarate_heatmap(args):
    # load configuration file 
    cfg = load_cfg_file(args.config_file)

    # init
    sample_rate = cfg.dataset.sample_rate
    clip_duration_ms = cfg.dataset.clip_duration_ms
    desired_samples = int(sample_rate * clip_duration_ms / 1000)

    # load model
    model = kws_load_model(cfg.general.save_dir, int(cfg.general.gpu_ids), cfg.test.model_epoch)
    net = model['prediction']['net']
    net.eval()
    
    # load data
    data, data_tensor = load_data(args, cfg, sample_rate, desired_samples)

    # save model input 
    data = data.reshape((-1, 40))
    plot_spectrogram(data.T, os.path.join(args.output_dir, 'fbank.png'))

    # infer
    data_tensor = data_tensor.cuda()
    score = net(data_tensor.unsqueeze(0), bool_draw_features=True, output_dir=args.output_dir)
    score = F.softmax(score, dim=1)
    score = score.cpu().data.numpy()
    print("Score: {}".format(score))
 

def main():
    default_config_file = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui1_6_res15_12162020/kws_config_xiaorui.py"
    default_input_wav = "/home/huanyuan/code/demo/Speech/KWS/visualization/audio/RM_KWS_XIAORUI_xiaorui_S012M1D00T001.wav"
    default_output_dir = "/home/huanyuan/code/demo/Speech/KWS/visualization/audio"
    parser = argparse.ArgumentParser(description='Streamax KWS Heatmap Engine')
    parser.add_argument('--config_file', type=str, default=default_config_file, help='config file')
    parser.add_argument('--input_wav', type=str, default=default_input_wav, help='input wav path')
    parser.add_argument('--output_dir', type=str, default=default_output_dir, help='output path')
    args = parser.parse_args()

    genarate_heatmap(args)


if __name__ == "__main__":
    main()