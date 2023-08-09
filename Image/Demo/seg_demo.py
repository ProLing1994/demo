import argparse
import cv2
import numpy as np
import os
import sys
import time 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.segmentation2d.rm_ai_mmseg.demo.segmentor.segmentor import Segmentor


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def inference_imgs(args):

    # mkdir 
    create_folder(args.output_img_dir)

    # model init 
    segmentor = Segmentor(args.seg_config, args.seg_checkpoint, device='cuda:0')

    # img init 
    img_list = np.array(os.listdir(args.img_dir))
    img_list = img_list[[img.endswith(args.suffix) for img in img_list]]
    img_list.sort()

    for idx in tqdm(range(len(img_list))):

        img_path = os.path.join(args.img_dir, img_list[idx])
        output_img_path = os.path.join(args.output_img_dir, img_list[idx])

        # img
        img = cv2.imread(img_path)

        # segmentor
        start = time.time()
        mask = segmentor.seg( img )
        end = time.time()
        print('Running time: %s Seconds'%(end-start))

        mask_img = np.zeros(img.shape, dtype=img.dtype)
        mask_img[:, :, 0] = mask[0][:, :] * 0
        mask_img[:, :, 1] = mask[0][:, :] * 0
        mask_img[:, :, 2] = mask[0][:, :] * 255

        # mask_img
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.3, gamma=0.)
        cv2.imwrite(output_img_path, mask_img)

        
def inference_video(args):

    # mkdir 
    create_folder(args.output_video_dir)

    # model init 
    segmentor = Segmentor(args.seg_config, args.seg_checkpoint, device='cuda:0')

    # video init 
    video_list = np.array(os.listdir(args.video_dir))
    video_list = video_list[[video.endswith(args.suffix) for video in video_list]]
    video_list.sort()

    for idx in tqdm(range(len(video_list))):
        video_path = os.path.join(args.video_dir, video_list[idx])

        cap = cv2.VideoCapture(video_path) 
        print(int(cap.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧

        output_video_path = os.path.join(args.output_video_dir, video_list[idx])
        create_folder(os.path.dirname(output_video_path))

        frame_idx = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), True)

        while True:
            ret, img = cap.read()

            if not ret: # if the camera over return false
                break

            # segmentor
            start = time.time()
            mask = segmentor.seg( img )
            end = time.time()
            print('Running time: %s Seconds'%(end-start))

            mask_img = np.zeros(img.shape, dtype=img.dtype)
            mask_img[:, :, 0] = mask[0][:, :] * 0
            mask_img[:, :, 1] = mask[0][:, :] * 0
            mask_img[:, :, 2] = mask[0][:, :] * 255

            # mask_img
            mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.3, gamma=0.)
            output_video.write(mask_img)

            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # seg pth
    args.seg_config = "/mnt/huanyuan/model/image/seg/mobilenet_v2_ch160_128x256_80k_rm_safeisland/20230717/mobilenet_v2_ch160_128x256_80k_rm_safeisland.py"
    args.seg_checkpoint =  "/mnt/huanyuan/model/image/seg/mobilenet_v2_ch160_128x256_80k_rm_safeisland/20230717/iter_1000000.pth"

    #######################
    # inference_imgs
    #######################

    # zd
    args.img_dir = "/mnt/huanyuan2/data/image/RM_C28_safeisland/test/"
    args.output_img_dir = "/mnt/huanyuan2/data/image/RM_C28_safeisland/test_res/"

    # args.suffix = '.jpg'
    # inference_imgs(args)

    #######################
    # inference_video
    #######################
    
    # zg
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/normal/20210220_20210419/C28_mini/BL2-BL3/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/BL2-BL3-202307-seg/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2023_0310_0412/2M_后向_20230310_0310_0412/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/2M_后向_20230310_0310_0412/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2023_0310_0412/2M_前向_20230310_0310_0412/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/2M_前向_20230310_0310_0412/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2023_0310_0412_01/2M_后向_20230310_0310_0412_01/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/2M_后向_20230310_0310_0412_01/"
    args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2023_0310_0412_01/2M_前向_20230310_0310_0412_01/"
    args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/2M_前向_20230310_0310_0412_01/"

    # args.suffix = '.avi'
    args.suffix = '.mp4'
    inference_video(args)