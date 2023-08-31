import argparse
import cv2
import numpy as np
import os
import sys
import time 
from tqdm import tqdm

# caffe_root = '/home/huanyuan/code/caffe/'
caffe_root = '/home/huanyuan/code/caffe_ssd-ssd-gpu/'
sys.path.insert(0, caffe_root+'python')
import caffe


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def inference_imgs(args):

    # mkdir 
    create_folder(args.output_img_dir)

    # img init 
    img_list = np.array(os.listdir(args.img_dir))
    img_list = img_list[[img.endswith(args.suffix) for img in img_list]]
    img_list.sort()

    for idx in tqdm(range(len(img_list))):

        img_path = os.path.join(args.img_dir, img_list[idx])
        output_img_path = os.path.join(args.output_img_dir, img_list[idx])

        # img
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resize = cv2.resize(img_gray, args.size, interpolation=cv2.INTER_LINEAR)
        img_resize = img_resize.astype(np.float32)

        # segmentor
        start = time.time()
        args.net.blobs['data'].data[...] = img_resize
        mask = args.net.forward()['decon6_out'][0][0] * 255
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        mask[np.where(mask < 100)] = 0
        mask[np.where(mask >= 100)] = 1
        end = time.time()
        print('Running time: %s Seconds'%(end-start))

        mask_img = np.zeros(img.shape, dtype=img.dtype)
        mask_img[:, :, 0] = mask[:, :] * 0
        mask_img[:, :, 1] = mask[:, :] * 0
        mask_img[:, :, 2] = mask[:, :] * 255

        # mask_img
        mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.3, gamma=0.)
        cv2.imwrite(output_img_path, mask_img)

        
def inference_video(args):

    # mkdir 
    create_folder(args.output_video_dir)

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
            
            # img
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resize = cv2.resize(img_gray, args.size, interpolation=cv2.INTER_LINEAR)
            img_resize = img_resize.astype(np.float32)

            # segmentor
            start = time.time()
            args.net.blobs['data'].data[...] = img_resize
            mask = args.net.forward()['decon6_out'][0][0] * 255
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
            mask[np.where(mask < 100)] = 0
            mask[np.where(mask >= 100)] = 1
            end = time.time()
            print('Running time: %s Seconds'%(end-start))

            mask_img = np.zeros(img.shape, dtype=img.dtype)
            mask_img[:, :, 0] = mask[:, :] * 0
            mask_img[:, :, 1] = mask[:, :] * 0
            mask_img[:, :, 2] = mask[:, :] * 255

            # mask_img
            mask_img = cv2.addWeighted(src1=img, alpha=0.8, src2=mask_img, beta=0.3, gamma=0.)
            output_video.write(mask_img)

            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # seg pth
    args.caffe_model = "/home/huanyuan/share/huanyuan/GAF/huanyuan/novt/Tanker/Tanker/vm_seg_tanker_0220_1chn/deploy.caffemodel"
    args.prototxt_file = "/home/huanyuan/share/huanyuan/GAF/huanyuan/novt/Tanker/Tanker/vm_seg_tanker_0220_1chn/deploy.prototxt"
    args.size = (256, 144)

    # model init 
    caffe.set_device(0)
    caffe.set_mode_gpu()
    # caffe.set_mode_cpu()
    args.net = caffe.Net(args.prototxt_file, args.caffe_model, caffe.TEST)
    args.net.blobs['data'].reshape(1, 1, args.size[1], args.size[0])   # N C H W

    #######################
    # inference_imgs
    #######################

    # zd
    args.img_dir = "/mnt/huanyuan2/data/image/HY_Tanker/test_video/算法误报_20230517/jpg/粤ABG105抓拍机00-230805-164306-164406-01p014000039/"
    args.output_img_dir = "/mnt/huanyuan/temp/pc_demo/HY_Tanker/test_jpg/算法误报_20230517/粤ABG105抓拍机00-230805-164306-164406-01p014000039/"

    args.suffix = '.jpg'
    inference_imgs(args)

    #######################
    # inference_video
    #######################
    
    args.video_dir = "/mnt/huanyuan2/data/image/HY_Tanker/test_video/算法误报_20230517/avi/"
    args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/HY_Tanker/test_video/算法误报_20230517/"

    # args.suffix = '.avi'
    # # args.suffix = '.mp4'
    # inference_video(args)