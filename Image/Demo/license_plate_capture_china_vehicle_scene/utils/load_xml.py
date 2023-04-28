import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
import xml.etree.ElementTree as ET


def load_xml(args):

    # video init 
    video_list = np.array(os.listdir(args.video_dir))
    video_list = video_list[[video.endswith(args.suffix) for video in video_list]]
    video_list.sort()

    for idx in tqdm(range(len(video_list))):
        video_name = video_list[idx]
        video_path = os.path.join(args.video_dir, video_name)

        xml_dir = os.path.join(args.xml_dir, video_name.replace(args.suffix, ''))
        xml_list = os.listdir(xml_dir)

        # init
        plate_dict = {}

        for xml_idx in tqdm(range(len(xml_list))):
            xml_name = xml_list[xml_idx]
            xml_path = os.path.join(xml_dir, xml_name)

            tree = ET.parse(xml_path)
            root = tree.getroot()

            for object in root.findall('object'):
                # name
                classname = str(object.find('name').text)
                
                # bbox
                if 'plate' in classname:
                    print(classname)
                    classname_list = classname.split('_')
                    
                    if len(classname_list) == 3:
                        id = classname_list[0]
                        plate_ocr = classname_list[-1]

                        plate_dict[id] = plate_ocr
            
        print(plate_dict)
        pkl_path = os.path.join(args.video_dir, video_name.replace(args.suffix, '.pkl'))
        f = open(pkl_path, 'wb')
        pickle.dump(plate_dict, f)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.video_dir = "/mnt/huanyuan/temp/智观数据/展会/demo_plate/"
    args.xml_dir = "/mnt/huanyuan/temp/智观数据/展会/demo_plate/plate_xml/"
    args.suffix = '.avi'    

    load_xml(args)