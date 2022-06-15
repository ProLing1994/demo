import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm


def check_video(args):
    video_list = np.array(os.listdir(args.input_dir))
    video_list = video_list[[video.endswith(args.suffix) for video in video_list]]

    for idx in tqdm(range(len(video_list))):
        video_path = os.path.join(args.input_dir, video_list[idx])

        cap = cv2.VideoCapture(video_path) 
        video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
        video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if video_width != check_size[0] and video_height != check_size[1]:
            print(video_path, video_width, video_height)
     

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.input_dir = "/mnt/huanyuan2/data/image/ZG_Face/bus_face_20210120/shenzhen/"      # 1920 * 1080
    # args.input_dir = "/mnt/huanyuan2/data/image/ZG_Face/shenzhen_video_0607/"             # 1280 * 720
    args.input_dir = "/mnt/huanyuan2/data/image/ZG_Face/shenzhen_video_0610/"               # 1920 * 1080
    args.suffix = '.mp4'
    check_size = [1920, 1080]

    check_video(args)