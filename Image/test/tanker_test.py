import argparse
from tqdm import tqdm
from Tanker.networks.build_model import buildmodel
from Tanker.utils.utils import AttrDict
import yaml
import torch
import numpy as np
import os
import cv2
from numpy import polyfit
cls_list=['Disable','Normal']
def ToLeft(a,b,p):
    #return a.x * b.y + b.x * p.y + p.x * a.y - b.y * p.x - a.y * b.x - a.x * p.y > 0.0
    return a[0] * b[1] + b[0] * p[1] + p[0] * a[1] - b[1] * p[0] - a[1] * b[0] - a[0] * p[1] > 0.0

def val_video(net,camera_position='top'):
    global best_val_loss
    roi_dict={'top':[(580,438),(724,433),(1279,465),(0,519)],'side':[(500,145),(694,148),(620,574),(236,576)]}
    print("========Val==========")
    save_video=False
    net.eval()
    video_root='G:/华油数据集/视频数据/1201'
    video_list = os.listdir(video_root)
    for video_name in video_list[:]:
        print(video_list.index(video_name), len(video_list))
        video_path = os.path.join(video_root, video_name)
        #video_path='G:/华油数据集/视频数据/1121/9027000000000000-221116-055428-055448-000003212680.avi'
        cap = cv2.VideoCapture(video_path)
        if(save_video):
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    #获取视频的宽度
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   #获取视频的高度
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #video_writer = cv2.VideoWriter('D:/python_script/Tanker/out.mp4', fourcc, fps,(width, height))
            video_writer = cv2.VideoWriter('D:/python_script/Tanker/'+video_name[:-3]+'mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                    25, (1280, 720))
        frame = -1
        skip_num = 1
        cls_id=0
        while (cap.isOpened()):
            ret, image = cap.read()
            if (not ret):
                break
            else:
                frame += 1
                if(frame%skip_num!=0):
                    continue

                #image=cv2.imread('G:/7118000000000000-221116-225400-225420-01p013000000_00000.jpg')
                if(0):
                    yuv=cv2.cvtColor(image,cv2.COLOR_BGR2YUV_I420)
                    yuv=cv2.resize(yuv,(256,216))
                    img=cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR_I420)
                else:
                    img=cv2.resize(image,(256,144),interpolation=cv2.INTER_NEAREST)
                #img=cv2.imread('G:/7118000000000000-221116-225400-225420-01p013000000_00000_resize.jpg')
                img_in=img[np.newaxis,...]
                img_in=torch.from_numpy(img_in)
                img_in=img_in.transpose(1,3).cuda()
                img_in=img_in.transpose(2,3)
                if(config.model.cls_branch):
                    seg_pred,cls_pred = net(img_in.float())
                    seg_pred=seg_pred.detach().cpu().numpy()
                    cls_pred=cls_pred.detach().cpu().numpy()
                    cls_id=np.argmax(cls_pred[0])
                else:
                    seg_pred = net(img_in.float())
                    seg_pred=seg_pred.detach().cpu().numpy()
                tmp_seg=seg_pred[0][0]*255
                tmp_seg[np.where(tmp_seg<100)]=0
                tmp_seg[np.where(tmp_seg >=100)]=1
                # cv2.imshow('a',tmp_seg)
                # cv2.waitKey(-1)
                element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # 形态学去噪
                tmp_seg = cv2.morphologyEx(tmp_seg, cv2.MORPH_OPEN, element)  # 闭运算去噪

                contours, hierarchy = cv2.findContours(tmp_seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                max_area=0
                max_cnt=None
                left_point = [1000, 1000]
                far_point = [1000, 1000]
                right_point = [0, 0]
                if(len(contours)):
                    for cnt in contours:
                        area=cv2.contourArea(cnt)
                        if(area>max_area):
                            max_area=area
                            max_cnt=cnt
                    x=0
                    y=0
                    if(0):
                        for i in range(max_cnt.shape[0]):
                            if(max_cnt[i,0,0]>5 and max_cnt[i,0,0]<250 and max_cnt[i,0,1]<140):
                                cv2.circle(img, (max_cnt[i,0,0],max_cnt[i,0,1]), 1, [0, 255, 0], 1)
                                if(max_cnt[i,0,1]<=far_point[1]):
                                    if(max_cnt[i,0,1]==far_point[1] and max_cnt[i,0,0]>far_point[0]):
                                        continue
                                    far_point[0]=max_cnt[i,0,0]
                                    far_point[1] = max_cnt[i, 0, 1]
                                if(max_cnt[i,0,0]<=left_point[0]):
                                    left_point[0]=max_cnt[i,0,0]
                                    left_point[1] = max_cnt[i, 0, 1]
                                if(max_cnt[i,0,0]>=right_point[0]):
                                    right_point[0]=max_cnt[i,0,0]
                                    right_point[1] = max_cnt[i, 0, 1]
                    else:
                        '''
                        算法目标：通过质心点，将右侧点和顶侧点区分开。（即使会损失部分点）
                        算法流程：
                        1、找到所有点的质心
                        2、通过质心找到两侧的点。右侧点在质心左下角，顶侧点在质心右上角
                        3、对于右侧点通过最小二乘获得一条直线。之所以不用这一条线作为报警判断线，目的是增加鲁棒性，防止右侧检测点波动导致拟合的线抖动剧烈。
                        4、对右侧点进行排序，从最底端的点开始，计算距离拟合直线的距离，当距离小于3（待定），即为右下角的点。
                        5、由于这条线是拟合出来的，必定会有这么一个点。
                        6、同理可以求顶端右侧的点
                        7、最后通过三角形方案，找出顶端左侧的点。
                        '''
                        center=max_cnt[:,0,:]
                        center=np.mean(center,axis=0)
                        side_points=[]
                        top_points=[]
                        angle_points=[]
                        for i in range(max_cnt.shape[0]):
                            if(max_cnt[i,0,0]>5 and max_cnt[i,0,0]<250 and max_cnt[i,0,1]<140):

                                if ((max_cnt[i, 0, 0])<center[0] and (max_cnt[i, 0, 1])>center[1]):#左下
                                    side_points.append([max_cnt[i, 0, 0],max_cnt[i, 0, 1]])
                                    cv2.circle(img, (max_cnt[i,0,0],max_cnt[i,0,1]), 1, [0, 0, 255], 2)
                                elif ((max_cnt[i, 0, 0])>center[0] and (max_cnt[i, 0, 1])<center[1]):#右上
                                    top_points.append([max_cnt[i, 0, 0],max_cnt[i, 0, 1]])
                                    cv2.circle(img, (max_cnt[i,0,0],max_cnt[i,0,1]), 1, [255, 0, 0], 2)
                                else:
                                    angle_points.append([max_cnt[i, 0, 0],max_cnt[i, 0, 1]])
                                    cv2.circle(img, (max_cnt[i, 0, 0], max_cnt[i, 0, 1]), 1, [0, 255, 0], 2)

                        side_points.sort(key=lambda x:x[1],reverse=True)
                        side_points=np.array(side_points)
                        coeff = polyfit(side_points[:,0], side_points[:,1], 1)
                        poly_line=[(int(-coeff[1]/coeff[0]),0),(int((144-coeff[1])/coeff[0]),144)]
                        cv2.line(img,poly_line[0],poly_line[1],[200,255,0])
                        A=coeff[0]
                        B=-1
                        C=coeff[1]
                        ABmod=(A**2+1)**0.5
                        for point in side_points:
                            dist=(A*point[0]+B*point[1]+C)/ABmod
                            if(dist<3):
                                left_point=point
                                break

                        top_points.sort(key=lambda x:x[0],reverse=True)
                        top_points=np.array(top_points)
                        coeff = polyfit(top_points[:,0], top_points[:,1], 1)
                        poly_line=[(0,int(coeff[1])),(256,int(coeff[0]*256+coeff[1]))]
                        cv2.line(img,poly_line[0],poly_line[1],[200,255,0])
                        A=coeff[0]
                        B=-1
                        C=coeff[1]
                        ABmod=(A**2+1)**0.5
                        for point in top_points:
                            dist=(A*point[0]+B*point[1]+C)/ABmod
                            if(dist<3):
                                right_point=point
                                break

                        A=(right_point[1]-left_point[1])/(right_point[0]-left_point[0])
                        B=-1
                        C=right_point[1]-A*right_point[0]
                        ABmod = (A ** 2 + 1) ** 0.5
                        far_point=[0,0]
                        max_dist=0
                        near_points=[]
                        near_thr=3
                        for point in angle_points:
                            dist=(A*point[0]+B*point[1]+C)/ABmod
                            if(1):#NMS mode
                                if(dist>(max_dist+near_thr)):#NMS
                                    max_dist=dist
                                    near_points.clear()
                                    near_points.append(point)
                                elif((max_dist-dist)<near_thr):
                                    near_points.append(point)
                            else:#simple mode
                                if(dist>max_dist):
                                    max_dist=dist
                                    far_point=point
                        if(len(near_points)):
                            near_points=np.array(near_points)
                            far_point=np.mean(near_points,axis=0)
                            far_point=far_point.astype(np.int)

                cv2.circle(img, (left_point[0],left_point[1]), 2, [255, 0, 255], 2)
                cv2.circle(img, (far_point[0],far_point[1]), 2, [255, 0, 255], 2)
                cv2.circle(img, (right_point[0],right_point[1]), 2, [255, 0, 255], 2)
                far_point[0]=far_point[0]*5#映射回原图尺寸
                far_point[1] = far_point[1] * 5
                left_point[0]=left_point[0]*5
                left_point[1] = left_point[1] * 5
                right_point[0]=right_point[0]*5
                right_point[1] = right_point[1] * 5
                # if(camera_position=='side'):
                #     roi=roi_dict[camera_position]
                #     line_k=(left_point[1]-far_point[1])/(left_point[0]-far_point[0])
                #     line_b=(left_point[1]-line_k*left_point[0])
                #     dist_left_top=(roi[0][1]-line_b)/line_k-roi[0][0]
                #     dist_left_bottom=(roi[3][1]-line_b)/line_k-roi[3][0]
                #     dist_right_top=(roi[1][1]-line_b)/line_k-roi[1][0]
                #     dist_right_bottom=(roi[2][1]-line_b)/line_k-roi[2][0]
                #     side_dist=abs(dist_right_top+dist_right_bottom)
                #     if(side_dist<150 and max_area<20000 and max_area>8000 and right_point[1]>far_point[1]):
                #         standard_falg=True
                #     else:
                #         standard_falg=False
                #     #print(dist_left_top,dist_left_bottom,dist_right_top,dist_right_bottom)
                #     #print(side_dist)
                # else:
                #
                #     roi=roi_dict[camera_position]
                #     limit_roi=[(roi[0][0],roi[0][1]-80),(roi[1][0],roi[1][1]-80),(roi[1][0],roi[1][1]+80),(roi[0][0],roi[0][1]+80)]
                #     inside_flag= ToLeft(limit_roi[0],limit_roi[1],far_point) and ToLeft(limit_roi[1],limit_roi[2],far_point) \
                #                  and ToLeft(limit_roi[2],limit_roi[3], far_point) and ToLeft(limit_roi[3],limit_roi[0],far_point)
                #     if (max_area<20000 and max_area>8000 and inside_flag):
                #         standard_falg=True
                #     else:
                #         standard_falg=False


                img=cv2.resize(img,(1280,720))
                # if(camera_position=='top'):
                #     cv2.line(img, limit_roi[0], limit_roi[1], [255, 0, 0], 4)
                #     cv2.line(img, limit_roi[1], limit_roi[2], [255, 0, 0], 4)
                #     cv2.line(img, limit_roi[2], limit_roi[3], [255, 0, 0], 4)
                #     cv2.line(img, limit_roi[3], limit_roi[0], [255, 0, 0], 4)
                #
                #
                # cv2.line(img,roi_dict[camera_position][0],roi_dict[camera_position][1],[255,255,0],4)
                # cv2.line(img, roi_dict[camera_position][1], roi_dict[camera_position][2], [255, 255, 0], 4)
                # cv2.line(img, roi_dict[camera_position][2], roi_dict[camera_position][3], [255, 255, 0], 4)
                # cv2.line(img, roi_dict[camera_position][3], roi_dict[camera_position][0], [255, 255, 0], 4)
                # if(standard_falg):
                #     cv2.putText(img, '[Standard Position]', (200, 100), 2, 1, [255, 0, 0], 2)
                # else:
                #     cv2.putText(img, '[Abnormal Position]', (200, 100), 2, 1, [255, 0, 0], 2)
                cv2.putText(img,cls_list[cls_id], (50, 100), 2, 1, [0, 255, 0], 2)
                if(save_video):
                    video_writer.write(img)
                cv2.imshow('a',img)
                cv2.waitKey(1)

        if(save_video):
            video_writer.release()

    return




def val_img(net,camera_position='top'):
    image=cv2.imread('D:/python_script/Tanker/imgset/1118_2.png')
    img=cv2.resize(image,(256,144))
    img_in=img[np.newaxis,...]
    img_in=torch.from_numpy(img_in)
    img_in=img_in.transpose(1,3).cuda()
    img_in=img_in.transpose(2,3)
    seg_pred = net(img_in.float())
    seg_pred=seg_pred.detach().cpu().numpy()
    tmp_seg=seg_pred[0][0]*255
    tmp_seg[np.where(tmp_seg<100)]=0
    tmp_seg[np.where(tmp_seg >=100)]=1
    contours, hierarchy = cv2.findContours(tmp_seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area=0
    max_cnt=None
    if(len(contours)):
        for cnt in contours:
            area=cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                max_cnt=cnt
        x=0
        y=0
        left_point=[1000,1000]
        far_point=[1000,1000]
        right_point=[0,0]
        for i in range(max_cnt.shape[0]):
            if(max_cnt[i,0,0]>5 and max_cnt[i,0,0]<250 and max_cnt[i,0,1]<140):
                cv2.circle(img, (max_cnt[i,0,0],max_cnt[i,0,1]), 1, [0, 255, 0], 1)
                if(max_cnt[i,0,1]<=far_point[1]):
                    if(max_cnt[i,0,1]==far_point[1] and max_cnt[i,0,0]>far_point[0]):
                        continue
                    far_point[0]=max_cnt[i,0,0]
                    far_point[1] = max_cnt[i, 0, 1]
                if(max_cnt[i,0,0]<=left_point[0]):
                    left_point[0]=max_cnt[i,0,0]
                    left_point[1] = max_cnt[i, 0, 1]
                if(max_cnt[i,0,0]>=right_point[0]):
                    right_point[0]=max_cnt[i,0,0]
                    right_point[1] = max_cnt[i, 0, 1]

    far_point[0]=far_point[0]*5#映射回原图尺寸
    far_point[1] = far_point[1] * 5
    left_point[0]=left_point[0]*5
    left_point[1] = left_point[1] * 5
    right_point[0]=right_point[0]*5
    right_point[1] = right_point[1] * 5
    img=cv2.resize(img,(1280,720))
    cv2.imshow('a',img)
    cv2.waitKey(-1)
    return

def val(net,val_loader,loss_func):
    global best_val_loss
    print("========Val==========")
    net.eval()
    val_loss = 0
    threshold=0.5
    progressbar = tqdm(range(len(val_loader)))
    save_id=0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            img=sample[0].to(device)
            segLabel=sample[1].to(device)
            img=img.transpose(1,3)
            seg_pred = net(img.float())
            seg_pred=seg_pred.transpose(1,3)
            seg_pred=seg_pred.squeeze(-1)
            seg_pred=seg_pred.detach().cpu().numpy()
            img=img.transpose(1,3)
            img=img.detach().cpu().numpy()
            segLabel=segLabel.detach().cpu().numpy()
            for i in range(img.shape[0]):
                tmp_img=img[i]
                tmp_seg=seg_pred[i]
                #tmp_seg[np.where(tmp_seg>=threshold)]=255
                #tmp_seg[np.where(tmp_seg<threshold)]=0
                tmp_seg=tmp_seg*255
                tmp_img[:,:,2][np.where(tmp_seg>100)]=255

                cv2.imwrite('D:/save_img/pic_{}.jpg'.format(save_id),tmp_img.astype(np.uint8))
                cv2.imwrite('D:/save_img/pic1_{}.jpg'.format(save_id),tmp_seg.astype(np.uint8))
                save_id+=1
    progressbar.close()
    print("------------------------\n")
    return val_loss


def bgr2nv21(bgr):
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    height = bgr.shape[0]
    width = bgr.shape[1]

    u = i420[height: height + height // 4, :]
    u = u.reshape((1, height // 4 * width))
    v = i420[height + height // 4: height + height // 2, :]
    v = v.reshape((1, height // 4 * width))
    uv = np.zeros((1, height // 4 * width * 2))
    uv[:, 0::2] = v
    uv[:, 1::2] = u
    uv = uv.reshape((height // 2, width))
    nv21 = np.zeros((height + height // 2, width))
    nv21[0:height, :] = i420[0:height, :]
    nv21[height::, :] = uv
    return nv21


def nv212bgr(bgr):
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    height = bgr.shape[0]
    width = bgr.shape[1]

    u = i420[height: height + height // 4, :]
    u = u.reshape((1, height // 4 * width))
    v = i420[height + height // 4: height + height // 2, :]
    v = v.reshape((1, height // 4 * width))
    uv = np.zeros((1, height // 4 * width * 2))
    uv[:, 0::2] = v
    uv[:, 1::2] = u
    uv = uv.reshape((height // 2, width))
    nv21 = np.zeros((height + height // 2, width))
    nv21[0:height, :] = i420[0:height, :]
    nv21[height::, :] = uv
    return nv21

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='D:/python_script/Tanker/config/tanker_novt.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('--local_rank', type=int, default=-1,help='node rank for distributed training')
    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    configfile = open(args.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    resize_shape = config.data.resize
    device = torch.device(config.training.device)

    # ------------ preparation ------------
    # Add model
    net=buildmodel(config.model)
    net = net.to(device)
    pretrained_dict = torch.load(config.training.load_model)
    pretrained_dict=pretrained_dict['net']
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:{}'.format(missed_params))
    print('loaded model{}'.format(config.training.load_model))
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict,strict=True)
    camera_position='side'
    # image=cv2.imread('G:/test_img/origin.jpg')
    # chip_resize=cv2.imread('G:/test_img/chip_resize.jpg')
    # img=cv2.resize(image,(256,144),interpolation=cv2.INTER_NEAREST)
    # img2 = cv2.resize(image, (256, 144), interpolation=cv2.INTER_LINEAR)
    # #yuv=cv2.cvtColor(image,cv2.COLOR_BGR2YUV_I420)
    # yuv=bgr2nv21(image)
    # yuv=yuv.astype(np.uint8)
    # yuv=cv2.resize(yuv,(256,216))
    # yuv=cv2.cvtColor(yuv,cv2.cv2.COLOR_YUV2BGR_NV21)
    # err=yuv-chip_resize
    # print(err.max())
    # cv2.imshow('a',img)
    # cv2.imshow('b',yuv)
    # cv2.imshow('c',err)
    # cv2.waitKey(-1)
    #
    val_video(net,camera_position)

