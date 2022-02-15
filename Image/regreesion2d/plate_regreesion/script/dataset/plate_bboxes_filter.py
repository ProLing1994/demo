import argparse
import numpy as np
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET


def bboxes_filter(args):
    # mkdir 
    if not os.path.isdir(args.output_xml_dir):
        os.makedirs(args.output_xml_dir)

    # xml init 
    xml_list = np.array(os.listdir(args.xml_dir))
    xml_list = xml_list[[xml.endswith('.xml') for xml in xml_list]]
    xml_list.sort()
    
    for idx in tqdm(range(len(xml_list))):
        xml_path = os.path.join(args.xml_dir, xml_list[idx])
        output_xml_path = os.path.join(args.output_xml_dir, xml_list[idx])

        # load car bboxes
        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

        # 标签检测和标签转换
        for obj in root.iter('object'):
            classname = obj.find('name').text.lower().strip()

            if classname in args.select_name_list:
                bbox_obj = obj.find('bndbox')

                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox_obj = []
                for _, pt in enumerate(pts):
                    cur_pt = int(float(bbox_obj.find(pt).text)) - 1
                    bndbox_obj.append(cur_pt)

                bndbox_height = bndbox_obj[3] - bndbox_obj[1] 

                # bndbox_height >= 20，设置为清晰标签 plate
                if bndbox_height >= 20:
                    obj.find('name').text = args.set_name_list[0]
                # bndbox_height >= 10，设置为非清晰标签 fuzzy_plate
                elif bndbox_height >= 10:
                    obj.find('name').text = args.set_name_list[1]
                # bndbox_height < 10，删除车牌标签
                else:
                    obj.find('name').text = args.set_name_list[2]

        # 删除无用标签
        for obj in root.findall('object'):
            classname = obj.find('name').text.lower().strip()

            if classname == args.set_name_list[2]:
                root.remove(obj)

        tree.write(output_xml_path)

    return 


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.xml_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone/XML_license_plate"
    args.output_xml_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone/XML_license_plate_filter"

    args.select_name_list = ["license_plate"]
    args.set_name_list = ["plate", "fuzzy_plate", "delete_plate"]
    bboxes_filter(args)


if __name__ == '__main__':
    main()
