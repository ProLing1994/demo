import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os 
import sys
from tqdm import tqdm


def plot_hist(data, bins, xlabel='', ylabel='', title='', savefig=''):
    # 绘制直方图
    plt.hist(x = data, # 指定绘图数据
            bins = bins, # 指定直方图中条块的个数
            color = 'steelblue', # 指定直方图的填充色
            edgecolor = 'steelblue' # 指定直方图的边框色
            )
    # 添加x轴和y轴标签
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # 添加标题
    plt.title(title)
    # 显示图形
    # plt.show()
    plt.savefig(savefig, dpi=300)
    plt.close()


def analysis_image_width_height(args):

    # image init 
    img_list = np.array(os.listdir(args.img_dir))
    img_list = img_list[[jpg.endswith('.jpg') for jpg in img_list]]
    img_list.sort()

    image_width_list = []
    image_height_list = []
    for idx in tqdm(range(len(img_list))):

        img_path = os.path.join(args.img_dir, img_list[idx])
        tqdm.write(img_path)

        img = cv2.imread(img_path)     

        image_width = img.shape[1]
        image_height = img.shape[0]

        tqdm_write = '{} {}'.format( image_width, image_height )
        tqdm.write(tqdm_write)

        image_width_list.append(image_width)
        image_height_list.append(image_height)

    plot_bins = max((int(np.array(image_width_list).max() - np.array(image_width_list).min())), 100)
    print( "max width: ", np.array(image_width_list).max(), ", min width: ", np.array(image_width_list).min())
    plot_hist(np.array(image_width_list), plot_bins, 'Image Width', 'frequency', 'Hist For Image Width', \
                os.path.join(args.output_dir, "hist_for_image_width_{}.png".format(args.name_format))) 

    plot_bins = max((int(np.array(image_height_list).max() - np.array(image_height_list).min())), 100)
    print( "max height: ", np.array(image_height_list).max(), ", min height: ", np.array(image_height_list).min())
    plot_hist(np.array(image_height_list), plot_bins, 'Image Height', 'frequency', 'Hist For Image Height', \
                os.path.join(args.output_dir, "hist_for_image_height_{}.png".format(args.name_format))) 


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # normal，林旭南提供数据
    # # args.name_format = "blue"
    # args.name_format = "green"
    # args.img_dir = os.path.join("/mnt/huanyuan2/data/image/LicensePlateRecognition/normal", args.name_format)
    # args.output_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/normal/"
    
    # # zg，智观加油站数据 2M
    # args.name_format = "blue"
    # # args.name_format = "green"
    # args.img_dir = os.path.join("/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_zhjyz_2M", args.name_format)
    # args.output_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_zhjyz_2M/"

    # zg，安徽淮北高速 5M
    args.name_format = "blue"
    args.img_dir = os.path.join("/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_ahhbgs_5M", args.name_format)
    args.output_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/zg_ahhbgs_5M/"
    
    analysis_image_width_height(args)


if __name__ == "__main__":
    main()