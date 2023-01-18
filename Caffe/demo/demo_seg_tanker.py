import cv2
import numpy as np 
from numpy import polyfit
import os
import sys
import time

caffe_root = '/home/huanyuan/code/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe


def RGBToNV12(image):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # y = (0.299 * r + 0.587 * g + 0.114 * b)
    # u = (-0.169 * r - 0.331 * g + 0.5 * b + 128)[::2, ::2]
    # v = (0.5 * r - 0.419 * g - 0.081 * b + 128)[::2, ::2]
    
    # y = 0.257 * r  + 0.504 * g + 0.098 * b + 16
    # u = (-0.148 * r - 0.291 * g + 0.439 * b + 128)[::2, ::2]
    # v = (0.439 * r - 0.368 * g - 0.071 * b + 128)[::2, ::2]

    y = (66.0/256.0 * r + 129.0/256.0 * g + 25.0/256.0 * b + 128.0/256.0) + 16
    u = ((-38.0/256.0 * r - 74.0/256.0 * g + 112.0/256.0 * b + 128.0/256.0) + 128)[::2, ::2]
    v = ((112.0/256.0 * r - 94.0/256.0 * g - 18.0/256.0 * b + 128.0/256.0) + 128)[::2, ::2]

    uv = np.zeros(shape=(u.shape[0], u.shape[1]*2))
    for i in range(0, u.shape[0]):
        for j in range(0, u.shape[1]):
            uv[i, 2*j] = u[i, j]
            uv[i, 2*j+1] = v[i, j]

    yuv = np.vstack((y, uv))
    return yuv.astype(np.uint8)


def test():
    # image_path = "/mnt/huanyuan2/data/image/HY_Tanker/test_video/test/test_side_00000.jpg"
    # image_path = "/mnt/huanyuan2/data/image/HY_Tanker/test_video/test/test_side_00010.jpg"
    # image_path = "/mnt/huanyuan2/data/image/HY_Tanker/算法误报_1121/jpg/top/7118000000000000-221117-073418-073438-01p014000000/7118000000000000-221117-073418-073438-01p014000000_00000.jpg"
    # image_path = "/mnt/huanyuan2/data/image/HY_Tanker/算法误报_1121/jpg/top/7118000000000000-221117-073418-073438-01p014000000/7118000000000000-221117-073418-073438-01p014000000_00010.jpg"
    # image_path = "/mnt/huanyuan2/data/image/HY_Tanker/算法误报_1121/jpg/side/7118000000000000-221116-225400-225420-01p013000000/7118000000000000-221116-225400-225420-01p013000000_00000.jpg"
    # image_path = "/mnt/huanyuan2/data/image/HY_Tanker/算法误报_1121/jpg/side/7118000000000000-221116-225400-225420-01p013000000/7118000000000000-221116-225400-225420-01p013000000_00010.jpg"
    # image_path = "/mnt/huanyuan2/data/image/HY_Tanker/算法误报_1121/jpg/side/0927000000000000-221117-073200-073220-01p013000000/0927000000000000-221117-073200-073220-01p013000000_00240.jpg"
    # image_path = "/mnt/huanyuan2/data/image/HY_Tanker/算法误报_1121/jpg/side/0927000000000000-221117-073200-073220-01p013000000/0927000000000000-221117-073200-073220-01p013000000_00250.jpg"
    # image_path = "/mnt/huanyuan2/data/image/HY_Tanker/算法误报_1121/jpg/side/0927000000000000-221117-073200-073220-01p013000000/0927000000000000-221117-073200-073220-01p013000000_00260.jpg"
    # image_path = "/mnt/huanyuan2/data/image/HY_Tanker/算法误报_1222/jpg/皖AH711800000000-221222-140157-140257-01p013000079/皖AH711800000000-221222-140157-140257-01p013000079_00160.jpg"
    image_path = "/mnt/huanyuan2/data/image/HY_Tanker/算法误报_20230101/jpg/img_01500.jpg"

    caffe_model = "/home/huanyuan/share/huanyuan/GAF/huanyuan/novt/Tanker/Tanker/tanker_1209.caffemodel"
    prototxt_file = "/home/huanyuan/share/huanyuan/GAF/huanyuan/novt/Tanker/Tanker/deploy.prototxt"
    size = (256, 144)

    # caffe.set_device(0)
    # caffe.set_mode_gpu()
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_file, caffe_model, caffe.TEST)
    net.blobs['data'].reshape(1, 3, size[1], size[0])   # N C H W

    img_origin = cv2.imread(image_path)

    # resize
    img_resize = cv2.resize(img_origin, size, interpolation=cv2.INTER_LINEAR)

    img = img_resize.astype(np.float32)
    img = img.transpose((2, 0, 1))
    net.blobs['data'].data[...] = img
    
    output = net.forward()['decon6_out'][0][0]
    output = output * 255

    output[np.where(output < 100)] = 0
    output[np.where(output >= 100)] = 1
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # 形态学去噪
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, element)  # 闭运算去噪

    img_resize[:,:,2][np.where(output == 1)] = 255
    output_path = os.path.join("/mnt/huanyuan2/data/image/HY_Tanker/test/", "caffe_{}.jpg".format(os.path.basename(image_path)[:-4]))
    cv2.imwrite(output_path, img_resize)

    contours, hierarchy = cv2.findContours(output.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    max_cnt = None
    left_point = [1000, 1000]
    far_point = [1000, 1000]
    right_point = [0, 0]

    if(len(contours)):

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if(area > max_area):
                max_area = area
                max_cnt = cnt

        '''
        算法目标：通过质心点，将左侧点和顶侧点区分开。（即使会损失部分点）
        算法流程：
        1、找到所有点的质心
        2、通过质心找到两侧的点。左侧点在质心左下角，顶侧点在质心右上角
        3、对于左侧点通过最小二乘获得一条直线。之所以不用这一条线作为报警判断线，目的是增加鲁棒性，防止左侧检测点波动导致拟合的线抖动剧烈。
        4、对左侧点进行排序，从最底端的点开始，计算距离拟合直线的距离，当距离小于3（待定），即为右下角的点。
        5、由于这条线是拟合出来的，必定会有这么一个点。
        6、同理可以求顶端左侧的点
        7、最后通过三角形方案，找出顶端左侧的点。
        '''
        center = max_cnt[:,0,:]
        center = np.mean(center, axis=0)            # 求平均
        side_points = []
        top_points = []
        angle_points = []
        for i in range(max_cnt.shape[0]):
            # if(max_cnt[i,0,0] > 5 and max_cnt[i,0,0] < 250 and max_cnt[i,0,1] < 140):  # 移除边界点

            #     if ((max_cnt[i, 0, 0])<center[0] and (max_cnt[i, 0, 1])>center[1]):#左下
            #         side_points.append([max_cnt[i, 0, 0],max_cnt[i, 0, 1]])
            #         cv2.circle(img_resize, (max_cnt[i,0,0],max_cnt[i,0,1]), 1, [0, 0, 255], 2)
            #     elif ((max_cnt[i, 0, 0])>center[0] and (max_cnt[i, 0, 1])<center[1]):#右上
            #         top_points.append([max_cnt[i, 0, 0],max_cnt[i, 0, 1]])
            #         cv2.circle(img_resize, (max_cnt[i,0,0],max_cnt[i,0,1]), 1, [255, 0, 0], 2)
            #     else:
            #         angle_points.append([max_cnt[i, 0, 0],max_cnt[i, 0, 1]])
            #         cv2.circle(img_resize, (max_cnt[i, 0, 0], max_cnt[i, 0, 1]), 1, [0, 255, 0], 2)

            if ((max_cnt[i, 0, 0])<center[0] and (max_cnt[i, 0, 1])>center[1]):#左下
                side_points.append([max_cnt[i, 0, 0],max_cnt[i, 0, 1]])
                cv2.circle(img_resize, (max_cnt[i,0,0],max_cnt[i,0,1]), 1, [0, 0, 255], 2)
            elif ((max_cnt[i, 0, 0])>center[0] and (max_cnt[i, 0, 1])<center[1]):#右上
                top_points.append([max_cnt[i, 0, 0],max_cnt[i, 0, 1]])
                cv2.circle(img_resize, (max_cnt[i,0,0],max_cnt[i,0,1]), 1, [255, 0, 0], 2)
            else:
                angle_points.append([max_cnt[i, 0, 0],max_cnt[i, 0, 1]])
                cv2.circle(img_resize, (max_cnt[i, 0, 0], max_cnt[i, 0, 1]), 1, [0, 255, 0], 2)

        side_points.sort(key = lambda x:x[1],reverse=True)
        side_points = np.array(side_points)
        coeff = polyfit(side_points[:, 0], side_points[:, 1], 1)
        poly_line=[(int(-coeff[1]/coeff[0]), 0), (int((144-coeff[1])/coeff[0]), 144)]
        cv2.line(img_resize, poly_line[0], poly_line[1], [200,255,0])
        A = coeff[0]
        B = -1
        C = coeff[1]
        ABmod = (A**2 + 1) ** 0.5
        for point in side_points:
            dist = abs(A*point[0] + B*point[1] + C) / ABmod
            if (dist < 3):
                left_point = point
                break

        top_points.sort(key = lambda x:x[0],reverse=True)
        top_points = np.array(top_points)
        coeff = polyfit(top_points[:,0], top_points[:,1], 1)
        poly_line = [(0, int(coeff[1])), (256, int(coeff[0]*256+coeff[1]))]
        cv2.line(img_resize, poly_line[0], poly_line[1], [200,255,0])
        A = coeff[0]
        B = -1
        C = coeff[1]
        ABmod = (A**2 + 1) ** 0.5
        for point in top_points:
            dist = abs(A*point[0] + B*point[1] + C) / ABmod
            if (dist < 3):
                right_point = point
                break

        A = (right_point[1] - left_point[1]) / (right_point[0] - left_point[0])
        B = -1
        C = right_point[1] - A * right_point[0]
        ABmod = (A ** 2 + 1) ** 0.5
        far_point = [0,0]
        max_dist = 0
        near_points = []
        near_thr = 3
        for point in angle_points:
            dist=(A*point[0]+B*point[1]+C)/ABmod
            if(1):#NMS mode
                if(dist > (max_dist + near_thr)):#NMS
                    max_dist = dist
                    near_points.clear()
                    near_points.append(point)
                elif((max_dist - dist) < near_thr):
                    near_points.append(point)
            else:#simple mode
                if(dist>max_dist):
                    max_dist=dist
                    far_point=point
        if(len(near_points)):
            near_points = np.array(near_points)
            far_point = np.mean(near_points,axis=0)
            far_point = far_point.astype(np.int)

    cv2.circle(img_resize, (left_point[0],left_point[1]), 2, [255, 0, 255], 2)
    cv2.circle(img_resize, (far_point[0],far_point[1]), 2, [255, 0, 255], 2)
    cv2.circle(img_resize, (right_point[0],right_point[1]), 2, [255, 0, 255], 2)

    # output_path = os.path.join("/mnt/huanyuan2/data/image/HY_Tanker/pc/", "caffe_res_{}.jpg".format(os.path.basename(image_path)[:-4]))
    output_path = os.path.join("/mnt/huanyuan2/data/image/HY_Tanker/test/", "caffe_res_{}.jpg".format(os.path.basename(image_path)[:-4]))
    cv2.imwrite(output_path, img_resize)


if __name__ == "__main__":
    test()
