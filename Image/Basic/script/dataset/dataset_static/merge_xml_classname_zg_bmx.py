import argparse
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.xml.xml_add import XmlAdder


def merge_classname(args):
    # mkdir 
    if not os.path.exists( args.output_xml_dir ):
        os.makedirs( args.output_xml_dir )
    
    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]

    for idx in tqdm(range(len(jpg_list))):
        xml_path = os.path.join(args.xml_dir, jpg_list[idx].replace(".jpg", ".xml"))
        merge_xml_path = os.path.join(args.merge_xml_dir, jpg_list[idx].replace(".jpg", ".xml"))
        output_xml_path = os.path.join(args.output_xml_dir, jpg_list[idx].replace(".jpg", ".xml"))
        
        # init bboxes
        bboxes = {}
        for classname in args.merge_select_name_list:
            bboxes[classname] = []
            bboxes[classname] = []
        
        # load bboxes
        tree = ET.parse(merge_xml_path)  # ET 是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()      # 获取根节点
        
        for obj in root.findall('object'):
            classname = obj.find('name').text.lower().strip()

            if classname in args.merge_select_name_list:
                bbox_obj = obj.find('bndbox')

                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox_obj = []
                for i, pt in enumerate(pts):
                    cur_pt = int(float(bbox_obj.find(pt).text)) - 1
                    bndbox_obj.append(cur_pt)
                
                bboxes[classname].append(bndbox_obj)
        
        # xml add
        xml_adder = XmlAdder(xml_path)
        for classname in args.merge_select_name_list:
            xml_adder.add_object(classname, bboxes[classname])
        
        xml_adder.write_xml(output_xml_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/ZG_BMX_detection/daminghu/"
    args.merge_input_dir = "/yuanhuan/data/image/ZG_BMX_detection/daminghu_head/"
    args.select_name_list = ["car", "bicycle", "motorcycle", "non-motorized", "non-motorized ", "person"]
    args.set_name_list = ["car", "bicycle", "motorcycle", "non-motorized", "non-motorized", "person"]
    args.merge_select_name_list =  ["head", "helmet"]
    args.merge_set_name_list = ["head", "helmet"]

    args.jpg_dir = args.input_dir + "JPEGImages/"
    args.xml_dir = args.input_dir + "XML/"
    args.merge_xml_dir = args.merge_input_dir + "XML/"
    args.output_xml_dir = args.input_dir + "XML_test/"

    merge_classname(args)