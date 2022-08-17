import argparse
import matplotlib.pyplot as plt
import numpy as np
import os 
from tqdm import tqdm
import xml.etree.ElementTree as ET


def plot_hist(data, bins, range, xlabel='', ylabel='', title='', savefig=''):
    # 绘制直方图
    plt.hist(x = data, # 指定绘图数据
            bins = bins, # 指定直方图的组距
            range= range,
            density=True, # 设置为频率直方图
            color = 'steelblue', # 指定直方图的填充色
            edgecolor = 'w', # 指定直方图的边框色
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


def analysis_label_width_height(args):

    # 遍历 label 
    for label_idx in tqdm(range(len(args.analysis_label_list))):
        analysis_label = args.analysis_label_list[label_idx]

        # 遍历训练集/测试集
        for dataset_type_idx in tqdm(range(len(args.dataset_type_list))):
            analysis_dataset_type = args.dataset_type_list[dataset_type_idx]

            # init
            label_width_list = []
            label_height_list = []
            
            # 遍历数据集
            for dataset_idx in tqdm(range(len(args.dataset_list))):
                analysis_dataset = args.dataset_list[dataset_idx]
                xml_dir = os.path.join(args.input_dir, analysis_dataset, args.xml_folder_name)
                analysis_file_txt = os.path.join(args.input_dir, analysis_dataset, "ImageSets/Main/", analysis_dataset_type + '.txt')

                # 加载数据列表
                with open(analysis_file_txt, "r") as f:
                    for line in tqdm(f):
                        xml_path = os.path.join(xml_dir, line.strip() + ".xml")

                        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
                        root = tree.getroot()   # 获取根节点

                        img_width = int(root.find('size').find('width').text)
                        img_height = int(root.find('size').find('height').text)

                        if img_width != args.image_width or img_height != args.image_height:
                            continue

                        for object in root.findall('object'):
                            # name
                            classname = str(object.find('name').text)

                            # bbox
                            bbox = object.find('bndbox')
                            pts = ['xmin', 'ymin', 'xmax', 'ymax']
                            bndbox = []
                            for i, pt in enumerate(pts):
                                cur_pt = int(float(bbox.find(pt).text)) - 1
                                bndbox.append(cur_pt)

                            bbox_width = bndbox[2] - bndbox[0]
                            bbox_height = bndbox[3] - bndbox[1]

                            if classname == analysis_label:

                                tqdm_write = '{}: {} {}'.format( classname, bbox_width, bbox_height )
                                tqdm.write(tqdm_write)

                                label_width_list.append(bbox_width)
                                label_height_list.append(bbox_height)

                print( "max width: ", np.array(label_width_list).max(), ", min width: ", np.array(label_width_list).min())
                plot_hist( np.array(label_width_list), args.plot_bins, args.width_thres, 'Image Width', 'frequency', 'Hist For Image Width', \
                            os.path.join(args.output_dir, "hist_for_image_width_{}_{}_{}.png".format('_'.join(args.dataset_list), '_'.join(args.analysis_label_list), '_'.join(args.dataset_type_list)))) 

                print( "max height: ", np.array(label_height_list).max(), ", min height: ", np.array(label_height_list).min())
                plot_hist( np.array(label_height_list), args.plot_bins, args.height_thres, 'Image Height', 'frequency', 'Hist For Image Height', \
                            os.path.join(args.output_dir, "hist_for_image_height_{}_{}_{}.png".format('_'.join(args.dataset_list), '_'.join(args.analysis_label_list), '_'.join(args.dataset_type_list)))) 


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/"
    args.xml_folder_name = "Annotations_CarBusTruckBicyclistMotorcyclistPerson"
    args.output_dir = "/yuanhuan/data/image/ZG_BMX_detection/analysis/"

    args.dataset_list = ['rongheng', 'rongheng_night_hongwai']
    # args.dataset_list = ['daminghu', 'daminghu_night']
    args.analysis_label_list = ['person']
    # args.dataset_type_list = ['trainval', 'test']
    args.dataset_type_list = ['test']

    args.image_width = 1920
    args.image_height = 1080

    args.width_thres = [0, 150]
    args.height_thres = [0, 300]
    args.plot_bins = 30

    analysis_label_width_height(args)

if __name__ == "__main__":
    main()