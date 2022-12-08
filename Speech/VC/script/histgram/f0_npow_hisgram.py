import argparse
import numpy as np
import os
from scipy.io import wavfile
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # noqa #isort:skip
import matplotlib.pyplot as plt  # isort:skip

from feature_extractor import FeatureExtractor


def create_histogram(data, figure_path, range_min=-70, range_max=20,
                     step=10, xlabel='Power [dB]'):
    """Create histogram
    Parameters
    ----------
    data : list,
        List of several data sequences
    figure_path : str,
        Filepath to be output figure
    range_min : int, optional,
        Minimum range for histogram
        Default set to -70
    range_max : int, optional,
        Maximum range for histogram
        Default set to -20
    step : int, optional
        Stap size of label in horizontal axis
        Default set to 10
    xlabel : str, optional
        Label of the horizontal axis
        Default set to 'Power [dB]'
    """

    # plot histgram
    plt.figure(figsize=(10, 5), dpi=200)
    plt.hist(data, bins=200, range=(range_min, range_max),
             density=True, histtype="stepfilled")
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.xticks(np.arange(range_min, range_max, step))

    figure_dir = os.path.dirname(figure_path)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plt.savefig(figure_path)
    plt.close()


def main():
    dcp = 'Create histogram for speaker-dependent configure'
    parser = argparse.ArgumentParser(description=dcp)
    parser.add_argument('-d', '--data_dir', type=str,  default="/mnt/huanyuan2/data/speech/asr/English/VCC2020-database/dataset/train/", help='data dir')
    parser.add_argument('-o', '--output_dir', type=str,  default="/mnt/huanyuan/data/speech/vc/English_dataset/dataset_audio_hdf5/VCC2020/conf_test/", help='output dir')
    # parser.add_argument('-d', '--data_dir', type=str,  default="/mnt/huanyuan2/data/speech/vc/Chinese/vc_test/train/", help='data dir')
    # parser.add_argument('-o', '--output_dir', type=str,  default="/mnt/huanyuan/data/speech/vc/Chinese_dataset/dataset_audio_hdf5/BZNSYP_Aishell3/conf_test/", help='output dir')
    args = parser.parse_args()

    spk_list = os.listdir(args.data_dir)
    spk_list.sort()

    for spk_idx in tqdm(range(len(spk_list))):
        spk_name = spk_list[spk_idx]
        spk_path = os.path.join(args.data_dir, spk_name)

        wav_list = os.listdir(spk_path)

        # init 
        f0s = []
        npows = []
        for wav_idx in range(len(wav_list)):
            wav_name = wav_list[wav_idx]

            # open waveform
            wavf = os.path.join(spk_path, wav_name)
            fs, x = wavfile.read(wavf)
            x = np.array(x, dtype=np.float)
            print("Extract: " + wavf)

            # constract FeatureExtractor class
            feat = FeatureExtractor(analyzer='world', fs=fs)

            # f0 and npow extraction
            f0, _, _ = feat.analyze(x)
            npow = feat.npow()

            f0s.append(f0)
            npows.append(npow)

        f0s = np.hstack(f0s).flatten()
        npows = np.hstack(npows).flatten()

        # create a histogram to visualize F0 range of the speaker
        f0histogrampath = os.path.join(args.output_dir, spk_name + '_f0histogram.png')
        create_histogram(f0s, f0histogrampath, range_min=40, range_max=700,
                        step=25, xlabel='Fundamental frequency [Hz]')

        # create a histogram to visualize npow range of the speaker
        npowhistogrampath = os.path.join(args.output_dir, spk_name + '_npowhistogram.png')
        create_histogram(npows, npowhistogrampath, range_min=-70, range_max=20,
                        step=5, xlabel="Frame power [dB]")


if __name__ == '__main__':
    main()