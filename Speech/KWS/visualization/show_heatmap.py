import argparse
import cv2
import glob
import librosa
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import sys

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.utils.train_tools import *

from KWS.impl.pred_pyimpl import kws_load_model
from KWS.impl.pred_pyimpl import audio_preprocess
from KWS.utils.train_tools import *
 

def load_data(input_wav, cfg, sample_rate, desired_samples):
    audio_data = librosa.core.load(input_wav, sr=sample_rate)[0]

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
    
    for idx in range(len(args.input_wav)):
        input_wav = args.input_wav[idx]

        # load data
        data, data_tensor = load_data(input_wav, cfg, sample_rate, desired_samples)
        
        # mkdir 
        output_dir = os.path.join(args.output_dir, os.path.basename(input_wav).split('.')[0])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # save model input 
        data = data.reshape((-1, 40))
        plot_spectrogram(data.T, os.path.join(output_dir, 'fbank.png'))

        # infer
        data_tensor = data_tensor.cuda()
        score = net(data_tensor.unsqueeze(0), bool_draw_features=True, output_dir=output_dir)
        score = F.softmax(score, dim=1)
        score = score.cpu().data.numpy()
        print("Score: {}".format(score))
 

def combine_heatmap(args):
    # collect heatmap
    heatmap_dict = {}
    for idx in range(len(args.input_wav)):
        input_wav_name =  os.path.basename(args.input_wav[idx]).split('.')[0]
        input_dir = os.path.join(args.output_dir, input_wav_name)
        heatmap_dict[input_wav_name] = {}

        file_type = '.png'
        file_list = glob.glob(os.path.join(input_dir, '*' + file_type))
        for file in file_list:
            heatmap_dict[input_wav_name][os.path.basename(file).split('.')[0]] = file
        file_list = glob.glob(os.path.join(input_dir, '*/*' + file_type))
        for file in file_list:
            file_name = os.path.basename(os.path.dirname(file)) + '_' + os.path.basename(file).split('.')[0]
            heatmap_dict[input_wav_name][file_name] = file

    # mkdir 
    output_dir = os.path.join(args.output_dir, os.path.basename(args.input_wav[0]).split('.')[0], "combine")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for key in tqdm(heatmap_dict[os.path.basename(args.input_wav[0]).split('.')[0]].keys()):

        fig = plt.figure()
        for wav_idx in range(len(args.input_wav)):

            input_wav_name =  os.path.basename(args.input_wav[wav_idx]).split('.')[0]
            file = heatmap_dict[input_wav_name][key]
            image = mpimg.imread(file)

            plt.subplot(len(heatmap_dict.keys()), 1, wav_idx + 1)
            plt.axis('off')
            plt.imshow(image)

        fig.savefig(os.path.join(output_dir, "{}.png".format(key)), dpi=600)
        fig.clf()
        plt.close()


def main():
    default_config_file = "/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui1_6_res15_12162020/kws_config_xiaorui.py"
    default_input_wav = ["/mnt/huanyuan/model/test/RM_Carchat_Mandarin_S004P00110.wav",
                            "/mnt/huanyuan/model/test/RM_Carchat_Mandarin_S004P00110_noise_reduction.wav"]
    default_output_dir = "/mnt/huanyuan/model/test"
    parser = argparse.ArgumentParser(description='Streamax KWS Heatmap Engine')
    parser.add_argument('--config_file', type=str, default=default_config_file, help='config file')
    parser.add_argument('--input_wav', type=str, default=default_input_wav, help='input wav path')
    parser.add_argument('--output_dir', type=str, default=default_output_dir, help='output path')
    args = parser.parse_args()

    # genarate_heatmap(args)
    combine_heatmap(args)

if __name__ == "__main__":
    main()