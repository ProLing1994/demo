import argparse
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.xml.xml_add import XmlAdder


def add_xml(args):
    # mkdir
    if not os.path.isdir(args.output_xml_dir):
        os.makedirs(args.output_xml_dir)
    
    # image init 
    img_list = np.array(os.listdir(args.img_dir))
    img_list = img_list[[jpg.endswith('.jpg') for jpg in img_list]]
    img_list = img_list[[os.path.exists(os.path.join(args.xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in img_list]]
    img_list.sort()

    for idx in tqdm(range(len(img_list))):
        img_path = os.path.join(args.img_dir, img_list[idx])
        xml_path = os.path.join(args.xml_dir, img_list[idx].replace(".jpg", ".xml"))
        add_xml_path = os.path.join(args.add_xml_dir, img_list[idx].replace(".jpg", ".xml"))
        output_xml_path = os.path.join(args.output_xml_dir, img_list[idx].replace(".jpg", ".xml"))

        tqdm.write(img_path)

        bboxes = {}
        bboxes[args.set_name] = []

        # load licenseplate bboxes
        tree = ET.parse(add_xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

        for obj in root.iter('object'):
            classname = obj.find('name').text.lower().strip()
            
            if classname == args.select_name:
                bbox_obj = obj.find('bndbox')

                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox_obj = []
                for i, pt in enumerate(pts):
                    cur_pt = int(float(bbox_obj.find(pt).text)) - 1
                    bndbox_obj.append(cur_pt)
                
                bboxes[args.set_name].append(bndbox_obj)

        # xml add
        xml_adder = XmlAdder(xml_path)
        xml_adder.add_object(args.set_name, bboxes[args.set_name])
        xml_adder.write_xml(output_xml_path)

def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.img_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_with_licenseplate/JPEGImages/"
    args.xml_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone/XML/"
    args.add_xml_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_with_licenseplate/XML/"
    args.output_xml_dir = "/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_with_licenseplate/XML_new/"
    args.select_name = 'licenseplate'
    args.set_name = 'plate'

    add_xml(args)


if __name__ == '__main__':
    main()
