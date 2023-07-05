import argparse
import cv2
import io
import lmdb
import numpy as np
import os
import sys
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def general_lmdb(lmdb_path, txt_file, txt_aug_file):
    assert lmdb_path.endswith('.lmdb'), "[ERROR] lmdb_path must end with 'lmdb'."
    if os.path.exists(lmdb_path):
        print("[Information] Folder [{:s}] already exists. Exit...".format(lmdb_path))
        return 
    
    # init
    commit_interval = 100

    with open(txt_file, "r") as f:
        txt_lines = f.readlines()

    with open(txt_aug_file, "r") as f:
        txt_aug_lines = f.readlines()
    
    txt_lines.extend(txt_aug_lines)

    # 估算映射空间大小（大概）
    line = txt_lines[0]
    line = line.strip()
    img_path = line.split(".jpg")[0] + '.jpg'
    city_mask_path = line.split(".jpg ")[1].split(".png")[0] + '.png'
    color_mask_path = line.split(".jpg ")[1].split(".png ")[1]
    img = cv2.imread(img_path)
    city_mask = cv2.imread(city_mask_path)
    color_mask = cv2.imread(color_mask_path)

    img_bytes = io.BytesIO()
    np.save(img_bytes, img)
    img_bytes = img_bytes.getvalue()

    city_mask_bytes = io.BytesIO()
    np.save(city_mask_bytes, city_mask)
    city_mask_bytes = city_mask_bytes.getvalue()
    
    color_mask_bytes = io.BytesIO()
    np.save(color_mask_bytes, color_mask)
    color_mask_bytes = color_mask_bytes.getvalue()
    
    data_size_img_mask = len(img_bytes) + len(city_mask_bytes) + len(color_mask_bytes)
    print('[Information] data size per load is: ', data_size_img_mask)
    data_size = data_size_img_mask * len(txt_lines)

    # map_size：
    # Maximum size database may grow to; used to size the memory mapping. If database grows larger
    # than map_size, an exception will be raised and the user must close and reopen Environment.
    env = lmdb.open(lmdb_path, map_size=data_size * 100)
    txn = env.begin(write=True)

    for idx in tqdm(range(len(txt_lines))):
        line = txt_lines[idx].strip()
        img_path = line.split(".jpg")[0] + '.jpg'
        city_mask_path = line.split(".jpg ")[1].split(".png")[0] + '.png'
        color_mask_path = line.split(".jpg ")[1].split(".png ")[1]
        
        # value
        img = cv2.imread(img_path)
        city_mask = cv2.imread(city_mask_path)
        color_mask = cv2.imread(color_mask_path)

        # put
        img_bytes = io.BytesIO()
        np.save(img_bytes, img)
        img_bytes = img_bytes.getvalue()
        txn.put(img_path.encode(), img_bytes)

        city_mask_bytes = io.BytesIO()
        np.save(city_mask_bytes, city_mask)
        city_mask_bytes = city_mask_bytes.getvalue()
        txn.put(city_mask_path.encode(), city_mask_bytes)

        color_mask_bytes = io.BytesIO()
        np.save(color_mask_bytes, color_mask)
        color_mask_bytes = color_mask_bytes.getvalue()
        txn.put(color_mask_path.encode(), color_mask_bytes)
        
        # load_bytes = io.BytesIO(img_bytes)
        # img_loads = np.load(load_bytes)

        if (idx + 1) % commit_interval == 0:
            txn.commit()
            # commit 之后需要再次 begin
            txn = env.begin(write=True)

    txn.commit()
    env.close()


def gen_seg_lmdb(args):

    print("Start preload audio({}): ".format(args.mode))

    # mkdir
    create_folder(args.output_dir)

    # init 
    lmdb_path = os.path.join(args.output_dir, '{}.lmdb'.format(args.mode))

    # general lmdb
    general_lmdb(lmdb_path, args.txt_file, args.txt_aug_file)

    print("Preload audio({}) Done!".format(args.mode))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="uae_20220804_0809") 
    parser.add_argument('--seg_name', type=str, default="seg_zd_202307") 
    parser.add_argument('--seg_lmdb_name', type=str, default="seg_zd_lmdb_202307") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/") 
    args = parser.parse_args()

    args.output_dir = os.path.join(args.input_dir, args.seg_lmdb_name, args.date_name)
    args.input_dir = os.path.join(args.input_dir, args.seg_name, args.date_name)

    print("gen seg lmdb.")
    print("input_dir: {}".format(args.input_dir))

    args.txt_file = os.path.join(args.input_dir, "city_color_label/ImageSets/Main/train.txt")
    args.txt_aug_file = os.path.join(args.input_dir, "city_color_label/ImageSets_aug/ImageSets/Main/train.txt")
    args.mode = 'train'
    gen_seg_lmdb(args)

    args.txt_file = os.path.join(args.input_dir, "city_color_label/ImageSets/Main/val.txt")
    args.txt_aug_file = os.path.join(args.input_dir, "city_color_label/ImageSets_aug/ImageSets/Main/val.txt")
    args.mode = 'val'
    gen_seg_lmdb(args)

    args.txt_file = os.path.join(args.input_dir, "city_color_label/ImageSets/Main/test.txt")
    args.txt_aug_file = os.path.join(args.input_dir, "city_color_label/ImageSets_aug/ImageSets/Main/test.txt")
    args.mode = 'test'
    gen_seg_lmdb(args)