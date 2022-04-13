import argparse
import json
import numpy as np
import os
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Image')
from Basic.script.xml.xml_write import write_xml


def json_xml(args):
    json_list = np.array(os.listdir(args.json_dir))
    json_list = json_list[[jpg.endswith('.json') for jpg in json_list]]

    for idx in tqdm(range(len(json_list))):
        # check json
        json_name = json_list[idx]
        json_path = os.path.join(args.json_dir, json_name)
        if not json_name in json_list:
            continue
        
        # read json
        with open(json_path, 'r', encoding='UTF-8') as fr:
            try:
                annotation = json.load(fr)
            except:
                print(json_path)
                continue

        weight = annotation['width']
        height = annotation['height']
        img_shape = [weight, height, 3]

        name = os.path.basename(annotation['img_name'])

        xml_bboxes = {}
        for track in annotation['shapes']:
            label = track['label']
            if track['attributes'][0]['value'] in ['green', 'yellow']:
                color = track['attributes'][0]['value']
                number = track['attributes'][1]['value']
            else:
                color = track['attributes'][1]['value']
                number = track['attributes'][0]['value']

            if not label == 'plate':
                continue

            xy = np.array(track['points'])
            x1 = int(xy[0])
            y1 = int(xy[1])
            x2 = int(xy[2])
            y2 = int(xy[3])

            xml_bboxes[number] = [[x1, y1, x2, y2]]
        
        img_path = os.path.join(args.jpg_dir, name)
        output_xml_path = os.path.join(args.xml_dir, str(name).replace('.jpg', '.xml'))
        write_xml(output_xml_path, img_path, xml_bboxes, img_shape)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.jpg_dir = "/mnt/huanyuan/temp/车牌测试样本/"
    args.json_dir = "/mnt/huanyuan/temp/车牌测试样本/"
    args.xml_dir = "/mnt/huanyuan/temp/车牌测试样本/"

    json_xml(args)


    
