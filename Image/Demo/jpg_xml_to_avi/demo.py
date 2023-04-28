import argparse
import cv2
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *


color_dict = {
                "face": (0, 0, 255),
            }

def NiceBox(img,rect,line_color,thickness=3,mask=True,mask_chn=2):
    width=rect[2]-rect[0]
    height=rect[3]-rect[1]
    line_len=max(10,min(width*0.15,height*0.15))
    line_len=int(line_len)
    cv2.line(img,(rect[0],rect[1]),(rect[0]+line_len,rect[1]),line_color,thickness=thickness)
    cv2.line(img,(rect[2]-line_len,rect[1]),(rect[2],rect[1]),line_color,thickness=thickness)
    cv2.line(img,(rect[0],rect[3]),(rect[0]+line_len,rect[3]),line_color,thickness=thickness)
    cv2.line(img,(rect[2]-line_len,rect[3]),(rect[2],rect[3]),line_color,thickness=thickness)
    cv2.line(img,(rect[0],rect[1]),(rect[0],rect[1]+line_len),line_color,thickness=thickness)
    cv2.line(img,(rect[0],rect[3]-line_len),(rect[0],rect[3]),line_color,thickness=thickness)
    cv2.line(img,(rect[2],rect[1]),(rect[2],rect[1]+line_len),line_color,thickness=thickness)
    cv2.line(img,(rect[2],rect[3]-line_len),(rect[2],rect[3]),line_color,thickness=thickness)
    if(mask):
        mask=np.zeros(img.shape[:2],dtype=np.uint8)
        coordinate=[[[rect[0],rect[1]],[rect[2],rect[1]],[rect[2],rect[3]],[rect[0],rect[3]]]]
        coordinate=np.array(coordinate)
        mask=cv2.fillPoly(mask,coordinate,100)
        mask_pos=mask.astype(np.bool)
        #mask_color=np.array(mask_color,dtype=np.uint8)
        img1=img[:,:,mask_chn]
        img1[np.where(mask!=0)] = mask[np.where(mask!=0)]
        img[:,:,mask_chn]=img1
    return img



def demo(args):

    # mkdir 
    create_folder(args.output_video_dir)
    
    file_list = os.listdir(args.jpg_dir)

    for idx in tqdm(range(len(file_list))):
        file_name = file_list[idx]
        file_path = os.path.join(args.jpg_dir, file_name)

        img_list = np.array(os.listdir(file_path))
        img_list = img_list[[img.endswith('.jpg') for img in img_list]]
        img_list.sort()

        # video
        video_path = os.path.join(args.output_video_dir, '{}.avi'.format(file_name))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(video_path, fourcc, 20.0, (2592, 1520), True)
        
        for idy in tqdm(range(len(img_list))):
            img_name = img_list[idy]
            img_path = os.path.join(file_path, img_name)
            xml_name = img_name.replace('.jpg', '.xml')

            # img
            img = cv2.imread(img_path)

            # xml
            xml_path = os.path.join(file_path, xml_name)
            if os.path.exists(xml_path):

                tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
                root = tree.getroot()   # 获取根节点

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

                    img = NiceBox(img, bndbox, color_dict["face"], thickness=3, mask=False)

                    img = cv2.putText(img, "face", (bndbox[0], bndbox[1] - 15), cv2.FONT_HERSHEY_COMPLEX, 3, color_dict["face"], 2)
            output_video.write(img)
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.jpg_dir = "/mnt/huanyuan/temp/智观数据/展会/demo_face/jpg/"
    args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/智观数据/展会/demo_face/"

    demo(args)