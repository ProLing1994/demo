import argparse
import numpy as np
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET


def select_classname(args):
    # 遍历数据集
    for dataset_idx in tqdm(range(len(args.dataset_list))):
        select_dataset = args.dataset_list[dataset_idx]

        anno_dir = os.path.join(args.input_dir, select_dataset, args.anno_name)
        select_file_txt = os.path.join(args.input_dir, select_dataset, "ImageSets/Main/", args.dataset_type + '.txt')

        for widtg_bar_idx in tqdm(range(len(args.width_bar_list))):
            select_widtg_bar = args.width_bar_list[widtg_bar_idx]
            select_widtg_bar_str = [str(idx) for idx in select_widtg_bar]
            out_xml_dir = os.path.join(args.output_dir, select_dataset, 'Annotations_' + args.select_label + '_' + '_'.join(select_widtg_bar_str))

            # mkdir 
            if not os.path.exists( out_xml_dir ):
                os.makedirs( out_xml_dir )
        
            # 加载数据列表
            with open(select_file_txt, "r") as f:
                for line in tqdm(f):
                    xml_path = os.path.join(anno_dir, line.strip() + ".xml")

                    tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
                    root = tree.getroot()   # 获取根节点


                    # 标签检测和标签转换
                    for object in root.findall('object'):
                        # name
                        classname = object.find('name').text.lower().strip()

                        # bbox
                        bbox = object.find('bndbox')
                        pts = ['xmin', 'ymin', 'xmax', 'ymax']
                        bndbox = []
                        for i, pt in enumerate(pts):
                            cur_pt = int(float(bbox.find(pt).text)) - 1
                            bndbox.append(cur_pt)

                        bbox_width = bndbox[2] - bndbox[0]
                        bbox_height = bndbox[3] - bndbox[1]

                        if classname == args.select_label:

                            if bbox_width >= select_widtg_bar[0] and bbox_width < select_widtg_bar[1]:
                                pass
                            else:
                                object.find('name').text = args.ignore_label
                        else:
                            root.remove(object)

                    output_xml_path = os.path.join(out_xml_dir, line.strip() + ".xml")
                    tree.write(output_xml_path)
                        

def select_classname_test_dataset(args):
    # 遍历数据集
    for dataset_idx in tqdm(range(len(args.dataset_list))):
        select_dataset = args.dataset_list[dataset_idx]
        print('Date name = {}'.format(select_dataset))

        anno_dir = os.path.join(args.input_dir, select_dataset + '_' + args.anno_name)

        for widtg_bar_idx in tqdm(range(len(args.width_bar_list))):
            select_widtg_bar = args.width_bar_list[widtg_bar_idx]
            select_widtg_bar_str = [str(idx) for idx in select_widtg_bar]

            out_xml_dir = os.path.join(args.output_dir, select_dataset + '_' 'Annotations_' + args.select_label + '_' + '_'.join(select_widtg_bar_str))

            # mkdir 
            if not os.path.exists( out_xml_dir ):
                os.makedirs( out_xml_dir )

            anno_list = np.array(os.listdir(anno_dir))
            anno_list = anno_list[[anno.endswith('.xml') for anno in anno_list]]
            anno_list.sort()

            for anno_idx in tqdm(range(len(anno_list))):
                xml_path = os.path.join(anno_dir, anno_list[anno_idx])

                tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
                root = tree.getroot()   # 获取根节点


                # 标签检测和标签转换
                for object in root.findall('object'):
                    # name
                    classname = object.find('name').text.lower().strip()

                    # bbox
                    bbox = object.find('bndbox')
                    pts = ['xmin', 'ymin', 'xmax', 'ymax']
                    bndbox = []
                    for i, pt in enumerate(pts):
                        cur_pt = int(float(bbox.find(pt).text)) - 1
                        bndbox.append(cur_pt)

                    bbox_width = bndbox[2] - bndbox[0]
                    bbox_height = bndbox[3] - bndbox[1]

                    if classname == args.select_label:

                        if bbox_width >= select_widtg_bar[0] and bbox_width < select_widtg_bar[1]:
                            pass
                        else:
                            object.find('name').text = args.ignore_label
                    else:
                        root.remove(object)

                output_xml_path = os.path.join(out_xml_dir, anno_list[anno_idx])
                tree.write(output_xml_path)


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # 测试集
    args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/"
    args.anno_name = "Annotations_CarBusTruckBicyclistMotorcyclistPerson"
    args.output_dir = "/yuanhuan/data/image/ZG_BMX_detection/"

    args.dataset_list = ['rongheng', 'rongheng_night_hongwai', 'daminghu', 'daminghu_night']
    args.dataset_type = 'test'

    args.select_label = 'person'
    args.ignore_label = "person_ignore"
    args.width_bar_list = [ [10, 20], [20, 30], [30, 40], [40, 50], [50, 60], [60, 70], [70, 80], [80, 120] ]
    
    select_classname(args)

    # 测试数据
    args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/"
    args.anno_name = "Annotations_CarBusTruckBicyclistMotorcyclistPerson_filter"
    args.output_dir = "/yuanhuan/data/image/ZG_BMX_detection/"

    args.dataset_list = ['banmaxian_test_image/2M_DaMingHu_far', 'banmaxian_test_image/2M_DaMingHu_near', 'banmaxian_test_image/2M_DaMingHu_night_far', 'banmaxian_test_image/2M_DaMingHu_night_near']
    # args.dataset_list = ['banmaxian_test_image/2M_RongHeng_far', 'banmaxian_test_image/2M_RongHeng_near', 'banmaxian_test_image/2M_RongHeng_night_far', 'banmaxian_test_image/2M_RongHeng_night_near']

    args.select_label = 'person'
    args.ignore_label = "person_ignore"
    args.width_bar_list = [ [10, 20], [20, 30], [30, 40], [40, 50], [50, 60], [60, 70], [70, 80], [80, 120] ]

    select_classname_test_dataset(args)

if __name__ == "__main__":

    main()
