import argparse
import cv2
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from sd.Paint-by-Example.scripts.inference import load_model_from_config

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *
from Basic.script.xml.xml_write import write_xml


def infer(args):

    # mkdir 
    create_folder(args.output_img_dir)
    create_folder(args.output_xml_dir)

    jpg_list = np.array(os.listdir(args.input_img_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.input_xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]
    jpg_list.sort()

    for idx in tqdm(range(len(jpg_list))):
    
        img_path = os.path.join(args.input_img_dir, jpg_list[idx])
        xml_path = os.path.join(args.input_xml_dir, jpg_list[idx].replace(".jpg", ".xml"))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="China") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_upspeed/original/") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/RM_upspeed/sd/paint_by_example") 
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.date_name)
    args.output_dir = os.path.join(args.output_dir, args.date_name)

    print("infer.")
    print("date_name: {}".format(args.date_name))
    print("input_dir: {}".format(args.input_dir))
    print("output_dir: {}".format(args.output_dir))

    args.input_img_dir = os.path.join(args.input_dir, 'JPEGImages')
    args.input_xml_dir = os.path.join(args.input_dir, 'Annotations')
    args.output_img_dir = os.path.join(args.output_dir, 'JPEGImages')
    args.output_xml_dir = os.path.join(args.output_dir, 'Annotations')

    args.infer_key_list = ['sign_height_c',
                          'sign_hand_c',
                          'sign_handb_c']
    # w, h
    args.mini_size = (130, 130)
    args.crop_size = (512, 512)

    # network
    args.plms = True
    args.ddim_eta = 0.0
    args.seed = 42
    args.scale = 10
    args.config = "/yuanhuan/code/demo/Image/sd/Paint-by-Example/configs/v1_upspeed.yaml"
    args.ckpt = "/yuanhuan/code/demo/Image/sd/Paint-by-Example/models/Paint-by-Example/2023-07-18T10-22-16_v1_upspeed/checkpoints/epoch=000039.ckpt"
    args.reference_path = "/yuanhuan/data/image/RM_upspeed/crop/Europe/references/0000000000000000-220723-181708-181710-000002655250-sn00048_upspeed_spain_40.jpg"

    # init network
    seed_everything(args.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    infer(args)