import argparse
import copy
import cv2
import math
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


def select_safeisland(args, img, mask, draw_img):

    # init 
    mask = mask[0]

    # res
    select_info = {}
    select_info['select_bool'] = False
    select_info['k'] = 0
    select_info['b'] = 0
    
    # param
    min_area_threshold = 200
    min_line_length_threshold = 80
    Candidate_num_threshold = 3
    theta_threshold = 60
    
    # 1、过滤 mask
    points_y, points_x = np.where(mask!=0)
    if args.direction_type == "Forward":
        capture_point_top = args.capture_points[1]
        capture_point_bottom = args.capture_points[0]
        args.filt_top_down_line_k = float(capture_point_top[1] - capture_point_bottom[1]) / float(capture_point_top[0] - capture_point_bottom[0] + 1e-5);
        args.filt_top_down_line_b = float(capture_point_top[1] - capture_point_top[0] * args.filt_top_down_line_k);

        for idx in range(len(points_x)):
            if ((points_y[idx] - args.filt_top_down_line_b) / (args.filt_top_down_line_k + 1e-5) - points_x[idx]) < 0:
                mask[points_y[idx], points_x[idx]] = 0

    else:
        capture_point_top = args.capture_points[6]
        capture_point_bottom = args.capture_points[7]
        args.filt_top_down_line_k = float(capture_point_top[1] - capture_point_bottom[1]) / float(capture_point_top[0] - capture_point_bottom[0] + 1e-5);
        args.filt_top_down_line_b = float(capture_point_top[1] - capture_point_top[0] * args.filt_top_down_line_k);

        capture_point_top = args.capture_points[0]
        capture_point_bottom = args.capture_points[7]
        args.filt_left_right_line_k = float(capture_point_top[1] - capture_point_bottom[1]) / float(capture_point_top[0] - capture_point_bottom[0] + 1e-5);
        args.filt_left_right_line_b = float(capture_point_top[1] - capture_point_top[0] * args.filt_left_right_line_k);

        for idx in range(len(points_x)):
            if ((points_y[idx] - args.filt_top_down_line_b) / (args.filt_top_down_line_k + 1e-5) - points_x[idx]) > 0:
                mask[points_y[idx], points_x[idx]] = 0

            if ((points_y[idx] - args.filt_left_right_line_b) / (args.filt_left_right_line_k + 1e-5) - points_x[idx]) > 0:
                mask[points_y[idx], points_x[idx]] = 0

    # draw_img
    draw_img[:, :, 0] = mask[:, :] * 0
    draw_img[:, :, 1] = mask[:, :] * 255
    draw_img[:, :, 2] = mask[:, :] * 0
    draw_img = cv2.addWeighted(src1=img, alpha=0.8, src2=draw_img, beta=0.3, gamma=0.)

    # 2、连通域
    mask = np.expand_dims(mask, axis=-1)
    mask = mask.astype(img.dtype)
    # numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # mask info list 
    mask_info_list = []     # [{'id': id, 'area': area, 'length': length, 'center_x': center_x, 'center_y': center_y, 'theta': theta, 'k': k, 'b': b},]
    # for idx in range(numLabels):
    for idx in range(len(contours)):
        
        # # 背景
        # if idx == 0:
        #     continue

        id = idx
        # area = stats[idx, cv2.CC_STAT_AREA]
        area = len(contours[idx])
        print(area)
        
        mask_info_list.append({'id': id, 'area': area, 'length': 0, 'center_x': 0, 'center_y': 0, 'theta': 0, 'k': 0, 'b': 0})

    # 3、候选参考系，至多 Candidate_num_threshold 个
    if len(mask_info_list) > Candidate_num_threshold:
        def sort_area(data):
            return data['area']
        
        mask_info_list.sort(key=sort_area, reverse=True)
        mask_info_list = mask_info_list[:Candidate_num_threshold]

    for idx in range(len(mask_info_list)):
        
        mask_info = mask_info_list[idx]

        # 4、提取中心线
        points = []
        # points_y, points_x = np.where(labels == mask_info['id'])
        points_y, points_x = contours[mask_info['id']][:, : ,1], contours[mask_info['id']][:, : ,0]
        points_y = (points_y).reshape((-1))
        points_x = (points_x).reshape((-1))
        for point_x in range(min(points_x), max(points_x)):
            if len(points_y[points_x==point_x]):
                points.append((point_x, int((min(points_y[points_x==point_x]) + max(points_y[points_x==point_x])) / 2 + 0.5)))
        # print(len(points))
        points = np.array(points)
        length = len(points)
        center_x = int((min(points_x) + max(points_x)) / 2 + 0.5)
        center_y = int((min(points_y) + max(points_y)) / 2 + 0.5)
        
        # 5、计算斜率和截距
        if area > min_area_threshold and len(points) > min_line_length_threshold:
            
            # ransac
            output = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            k = output[1] / output[0]
            b = output[3] - k * output[2]

            # 6、根据斜率求夹角
            theta = int(math.fabs(np.arctan((args.capture_line_k-k)/(float(1 + args.capture_line_k*k)))*180/np.pi)+0.5)

            if theta < theta_threshold:
                mask_info_list.append({'id': id, 'area': area, 'length': length, 'center_x': center_x, 'center_y': center_y, 'theta': theta, 'k': k, 'b': b})

            # draw_img
            for point_idx in range(len(points)):
                cv2.circle(draw_img, points[point_idx], 1, (255,0,0), 1)

    # 7、sort 排序，挑选位置符合预期的参考平面（y值越小越好）
    if len(mask_info_list) > 0:
        def sort_location(data):
            return data['center_y']

        def sort_theta(data):
            return data['theta']
        
        mask_info_list.sort(key=sort_location)

        select_info['select_bool'] = True
        select_info['k'] = mask_info_list[0]['k']
        select_info['b'] = mask_info_list[0]['b']

        # # draw_img
        # start_x = 0
        # start_y = k * start_x + b
        # if start_y <= 0:
        #     start_y = 0
        #     start_x = ( - b) / k
        
        # end_x = img.shape[1] - 1
        # end_y = k * end_x + b
        # if end_y >= img.shape[0] - 1:
        #     end_y = img.shape[0] - 1
        #     end_x = (end_y - b) / k

        # cv2.line(draw_img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0,0,255), 5)
    
    # 绘制抓拍线
    start_x = 0
    start_y = args.capture_line_k * start_x + args.capture_line_b
    if start_y <= 0:
        start_y = 0
        start_x = ( - args.capture_line_b) / args.capture_line_k
    
    end_x = img.shape[1] - 1
    end_y = args.capture_line_k * end_x + args.capture_line_b
    if end_y >= img.shape[0] - 1:
        end_y = img.shape[0] - 1
        end_x = (end_y - args.capture_line_b) / args.capture_line_k

    # # draw_img
    # cv2.line(draw_img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255,255,255), 1)
    # for capture_point_idx in range(len(args.capture_points)):
    #     cv2.circle(draw_img, args.capture_points[capture_point_idx], 3, (255,255,255), 5)

    return select_info, draw_img


def stabel_safeisland(args, selec_img, select_info, stabel_info, draw_img):

    # select_info['select_bool'] = False
    # select_info['k'] = 0
    # select_info['b'] = 0

    # stabel_info['stabel_bool'] = False
    # stabel_info['cnt'] = 0
    # stabel_info['lost_cnt'] = 0
    # stabel_info['k'] = 0
    # stabel_info['b'] = 0

    if select_info['select_bool']:
        
        if stabel_info['cnt'] == 0:
            
            stabel_info['stabel_bool'] = False
            stabel_info['k'] = select_info['k']
            stabel_info['b'] = select_info['b']
            stabel_info['cnt'] = 1
            stabel_info['lost_cnt'] = 0

        elif stabel_info['cnt'] < 10:
            
            stabel_info['stabel_bool'] = False
            stabel_info['k'] = 0.99 * stabel_info['k'] + 0.01 * select_info['k']
            stabel_info['b'] = 0.99 * stabel_info['b'] + 0.01 * select_info['b']
            stabel_info['cnt'] += 1
            stabel_info['lost_cnt'] = 0

        elif stabel_info['cnt'] >= 10:
            
            stabel_info['stabel_bool'] = True
            stabel_info['k'] = 0.99 * stabel_info['k'] + 0.01 * select_info['k']
            stabel_info['b'] = 0.99 * stabel_info['b'] + 0.01 * select_info['b']
            stabel_info['cnt'] = 10
            stabel_info['lost_cnt'] = 0
    else:

        if stabel_info['cnt'] == 0:
            
            stabel_info['stabel_bool'] = False
            stabel_info['k'] = 0
            stabel_info['b'] = 0
            stabel_info['cnt'] = 0
            stabel_info['lost_cnt'] = 0

        elif stabel_info['lost_cnt'] >= 25:
    
            stabel_info['stabel_bool'] = False
            stabel_info['k'] = 0
            stabel_info['b'] = 0
            stabel_info['cnt'] = 0
            stabel_info['lost_cnt'] = 0

        else:
            
            stabel_info['lost_cnt'] += 1

    # draw_img
    draw_img = copy.deepcopy(selec_img)

    if stabel_info['stabel_bool']:

        k = stabel_info['k']
        b = stabel_info['b']

        start_x = 0
        start_y = k * start_x + b
        if start_y <= 0:
            start_y = 0
            start_x = ( - b) / k
        
        end_x = selec_img.shape[1] - 1
        end_y = k * end_x + b
        if end_y >= selec_img.shape[0] - 1:
            end_y = selec_img.shape[0] - 1
            end_x = (end_y - b) / k

        # draw_img
        cv2.line(draw_img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0,0,255), 5)

    return stabel_info, draw_img


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

        # select_safeisland
        start = time.time()
        selec_img = np.zeros(img.shape, dtype=img.dtype)
        select_info, selec_img = select_safeisland(args, img, mask, selec_img)
        end = time.time()
        print('Select Safeisland time: %s Seconds'%(end-start))
        
        # # stabel_safeisland
        # start = time.time()
        # stabel_img = np.zeros(img.shape, dtype=img.dtype)
        # stabel_info = {}
        # stabel_info['stabel_bool'] = False
        # stabel_info['cnt'] = 0
        # stabel_info['lost_cnt'] = 0
        # stabel_info['k'] = 0
        # stabel_info['b'] = 0
        # stabel_info, stabel_img = stabel_safeisland(args, selec_img, select_info, stabel_info, stabel_img)
        # end = time.time()
        # print('Stabel Safeisland time: %s Seconds'%(end-start))
        
        # write
        cv2.imwrite(output_img_path, selec_img)


def inference_video(args):

    # mkdir 
    create_folder(args.output_video_dir)

    # model init 
    segmentor = Segmentor(args.seg_config, args.seg_checkpoint, device='cuda:0')

    # video init 
    video_list = np.array(os.listdir(args.video_dir))
    video_list = video_list[[video.endswith(args.suffix) for video in video_list]]
    video_list.sort()

    # init (test)
    stabel_info = {}
    stabel_info['stabel_bool'] = False
    stabel_info['cnt'] = 0
    stabel_info['lost_cnt'] = 0
    stabel_info['k'] = 0
    stabel_info['b'] = 0

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
        output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2 ), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)), True)

        # init
        # stabel_info = {}
        # stabel_info['stabel_bool'] = False
        # stabel_info['cnt'] = 0
        # stabel_info['lost_cnt'] = 0
        # stabel_info['k'] = 0
        # stabel_info['b'] = 0

        while True:
            ret, img = cap.read()

            if not ret: # if the camera over return false
                output_video.release()
                cap.release()
                break

            # segmentor
            start = time.time()
            mask = segmentor.seg( img )
            end = time.time()
            print('Running time: %s Seconds'%(end-start))

            # select_safeisland
            start = time.time()
            selec_img = np.zeros(img.shape, dtype=img.dtype)
            select_info, selec_img = select_safeisland(args, img, mask, selec_img)
            end = time.time()
            print('Select Safeisland time: %s Seconds'%(end-start))
            
            # stabel_safeisland
            start = time.time()
            stabel_img = np.zeros(img.shape, dtype=img.dtype)
            stabel_info, stabel_img = stabel_safeisland(args, selec_img, select_info, stabel_info, stabel_img)
            end = time.time()
            print('Stabel Safeisland time: %s Seconds'%(end-start))

            # write
            # output_video.write(selec_img)
            stabel_img = cv2.resize(stabel_img, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2 )))
            output_video.write(stabel_img)

            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # seg pth
    args.seg_config = "/mnt/huanyuan/model/image/seg/mobilenet_v2_ch160_128x256_80k_rm_safeisland/20230717/mobilenet_v2_ch160_128x256_80k_rm_safeisland.py"
    args.seg_checkpoint =  "/mnt/huanyuan/model/image/seg/mobilenet_v2_ch160_128x256_80k_rm_safeisland/20230717/iter_1000000.pth"


    args.direction_type = "Forward"
    # args.direction_type = "Backward"
    if args.direction_type == "Forward":
        args.capture_points = [(965, 156),
                        (1312, 185), (1381, 238), (1434, 287), (1532, 372), (1733, 543), (1775, 585),
                        (1547, 701), (1489, 666), (1189, 362), (1077, 254), (1018, 208), 
                        (1083, 165), (1143, 219), (1192, 260), (1300, 364), (1587, 618), (1627, 658)]
    else:
        args.capture_points = [(1217, 141),
                        (1799, 259), (1744, 286), (1619, 347), (1405, 454), (837, 740), (199, 1065),
                        (210, 445), (491, 360), (774, 271), (987, 211), (1128, 170), 
                        (1629, 223), (1567, 252), (1438, 307), (1218, 403), (727, 619), (202, 849)]

    # y = kx + b
    capture_point_top = args.capture_points[5]
    capture_point_bottom = args.capture_points[8]
    args.capture_line_k = float(capture_point_top[1] - capture_point_bottom[1]) / float(capture_point_top[0] - capture_point_bottom[0] + 1e-5);
    args.capture_line_b = float(capture_point_top[1] - capture_point_top[0] * args.capture_line_k);
   
    #######################
    # inference_imgs
    #######################
    # zd
    args.img_dir = "/mnt/huanyuan2/data/image/RM_C28_safeisland/test/"
    args.output_img_dir = "/mnt/huanyuan2/data/image/RM_C28_safeisland/test_res/"

    args.suffix = '.jpg'
    inference_imgs(args)

    #######################
    # inference_video
    #######################

    # zg
    args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2023_0310_0412/2M_前向_20230310_0310_0412/"
    args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/2M_前向_20230310_0310_0412/"

    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2023_0310_0412_01/2M_前向_20230310_0310_0412_01/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/2M_前向_20230310_0310_0412_01/"

    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2021_1024_1025/2M_前向_2021_1024_1025/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/2M_前向_2021_1024_1025/"

    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2023_0109/2M_前向_2023_0109/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/2M_前向_2023_0109/"

    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2023_0215/2M_前向_2023_0215/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/2M_前向_2023_0215/"

    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/前向_special/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/前向_special/"

    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/normal/20210220_20210419/C28_mini/BL2-BL3/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/BL2-BL3-202307-seg/"

    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2023_0310_0412/2M_后向_20230310_0310_0412/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/2M_后向_20230310_0310_0412/"

    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2023_0310_0412_01/2M_后向_20230310_0310_0412_01/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/2M_后向_20230310_0310_0412_01/"

    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2021_1024_1025/2M_后向_2021_1024_1025/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/2M_后向_2021_1024_1025/"

    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2023_0109/2M_后向_2023_0109/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/2M_后向_2023_0109/"

    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/2023_0215/2M_后向_2023_0215/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/2M_后向_2023_0215/"

    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/SchBus_ZD_AD_C27_C28/safeisland/后向_special/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/90038/safeisland/后向_special/"


    # args.suffix = '.avi'
    args.suffix = '.mp4'
    # inference_video(args)