import argparse
import librosa
import os 
import sys

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *


def plot_hist(data, bins, xlabel='', ylabel='', title='', savefig=''):
    # 绘制直方图
    plt.hist(x = data, # 指定绘图数据
            bins = bins, # 指定直方图中条块的个数
            color = 'steelblue', # 指定直方图的填充色
            edgecolor = 'black' # 指定直方图的边框色
            )
    # 添加x轴和y轴标签
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # 添加标题
    plt.title(title)
    # 显示图形
    # plt.show()
    plt.savefig(savefig, dpi=300)


def analysis_audio_length(config_file):
    # load configuration file
    cfg = load_cfg_file(config_file)

    # init 
    sample_rate = cfg.dataset.sample_rate
    data_dir = cfg.general.data_dir
    positive_label = cfg.dataset.label.positive_label[0]
    input_dir = os.path.join(data_dir, positive_label)
    output_dir = os.path.join(cfg.general.data_dir, '../dataset_{}_{}'.format(
        cfg.general.version, cfg.general.date), 'data_distribution')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    file_list = os.listdir(input_dir)
    audio_length_list = []
    for file_name in tqdm(file_list):
        file_path = os.path.join(input_dir, file_name)
        audio_data = librosa.core.load(file_path, sr=sample_rate)[0]
        audio_length = int(len(audio_data) * 1000 / sample_rate)
        audio_length_list.append(audio_length)
        # if audio_length > 4000:
        #     print(file_path)

    plot_bins = max((int((np.array(audio_length_list).max() - np.array(audio_length_list).min()) // 1000.0) + 1) * 2, 6)
    plot_hist(np.array(audio_length_list), plot_bins, 'Audio Length', 'frequency', 'Hist For Audio Length', os.path.join(output_dir, "hist_for_audio_length.png"))   

    
def main():
    # Calculate the distribution of speech length of positive samples
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py")
    # parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaole.py")
    parser.add_argument('--config_file', type=str,  default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui.py")
    args = parser.parse_args()

    print("[Begin] Analysis Audio Length")
    analysis_audio_length(args.config_file)
    print("[Done] Analysis Audio Length")


if __name__ == "__main__":
    main()