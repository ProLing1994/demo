import argparse
from moviepy.editor import *
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *

def concatenate_videos( args ):

    # 定义一个数组
    videoclips_list = []

    # image init 
    video_list = np.array(os.listdir(args.video_dir))
    video_list = video_list[[video.endswith(args.suffix) for video in video_list]]
    video_list.sort()
    print(video_list)

    # mkdir 
    output_capture_path = os.path.join(args.output_video_dir, "target.mp4")
    create_folder(os.path.dirname(output_capture_path))
    
    for idx in tqdm(range(len(video_list))):
        video_path = os.path.join(args.video_dir, video_list[idx])

        # 载入视频
        video = VideoFileClip(video_path)

        # 添加到数组
        videoclips_list.append(video)

    # 拼接视频
    final_clip = concatenate_videoclips(videoclips_list)

    # 生成目标视频文件
    final_clip.to_videofile(output_capture_path, fps=30, remove_temp=False)

def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_ZHJYZ_220119/264原始视频/2M/"
    # args.output_video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_ZHJYZ_220119/264原始视频/2M_all/"
    args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_ZHJYZ_220119/264原始视频/2M_调试视频/"
    args.output_video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/jiayouzhan_test_video/ZG_ZHJYZ_220119/264原始视频/2M_调试视频_all/"
    args.suffix = '.avi'

    concatenate_videos( args )


if __name__ == '__main__':
    main()