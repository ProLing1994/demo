import cv2
import numpy as np 
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
    image_path = "/mnt/huanyuan2/data/image/HY_Tanker/算法误报_1121/jpg/side/0927000000000000-221117-073200-073220-01p013000000/0927000000000000-221117-073200-073220-01p013000000_00260.jpg"

    caffe_model = "/home/huanyuan/share/huanyuan/GAF/huanyuan/novt/Tanker/Tanker/tanker_1125.caffemodel"
    prototxt_file = "/home/huanyuan/share/huanyuan/GAF/huanyuan/novt/Tanker/Tanker/deploy.prototxt"
    size = (256, 144)

    # caffe.set_device(0)
    # caffe.set_mode_gpu()
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_file, caffe_model, caffe.TEST)
    net.blobs['data'].reshape(1, 3, size[1], size[0])   # N C H W

    img_origin = cv2.imread(image_path)

    # resize
    # img_origin = cv2.resize(img_origin, size, interpolation=cv2.INTER_NEAREST)
    img_origin = cv2.resize(img_origin, size, interpolation=cv2.INTER_LINEAR)
    # img_origin = cv2.resize(img_origin, size, interpolation=cv2.INTER_AREA)
    # img_origin = cv2.resize(img_origin, size, interpolation=cv2.INTER_CUBIC)
    # img_origin = cv2.resize(img_origin, size, interpolation=cv2.INTER_LANCZOS4)

    # # resize
    # # yuv = cv2.cvtColor(img_origin, cv2.COLOR_RGB2YUV_YV12)
    # yuv = RGBToNV12(img_origin)
    # # yuv = cv2.resize(yuv, (256, 216), interpolation=cv2.INTER_NEAREST)
    # # yuv = cv2.resize(yuv, (256, 216), interpolation=cv2.INTER_LINEAR)
    # # yuv = cv2.resize(yuv, (256, 216), interpolation=cv2.INTER_AREA)
    # # yuv = cv2.resize(yuv, (256, 216), interpolation=cv2.INTER_CUBIC)
    # yuv = cv2.resize(yuv, (256, 216), interpolation=cv2.INTER_LANCZOS4)
    # img_origin = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV21)

    output_path = os.path.join("/mnt/huanyuan2/data/image/HY_Tanker/pc/", "resize_img_{}.jpg".format(os.path.basename(image_path)[:-4]))
    cv2.imwrite(output_path, img_origin)
    cv_file = cv2.FileStorage(os.path.join("/mnt/huanyuan2/data/image/HY_Tanker/pc/", "resize_img_{}.xml".format(os.path.basename(image_path)[:-4])), cv2.FILE_STORAGE_WRITE)
    cv_file.write('test', img_origin)
    cv_file.release()

    img = img_origin.astype(np.float32)
    img = img.transpose((2, 0, 1))
    net.blobs['data'].data[...] = img
    
    output = net.forward()['decon6_out'][0][0]
    output = output * 255
    print(output)
    cv_file = cv2.FileStorage(os.path.join("/mnt/huanyuan2/data/image/HY_Tanker/pc/", "caffe_res_{}.xml".format(os.path.basename(image_path)[:-4])), cv2.FILE_STORAGE_WRITE)
    cv_file.write('test', output)
    cv_file.release()

    img_origin[:,:,2][np.where(output > 100)] = 255
    contours, hierarchy = cv2.findContours(output.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    output_path = os.path.join("/mnt/huanyuan2/data/image/HY_Tanker/pc/", "caffe_res_{}.jpg".format(os.path.basename(image_path)[:-4]))
    cv2.imwrite(output_path, img_origin)


def load_novt_xml_test():
    # image_path = "/mnt/huanyuan2/data/image/HY_Tanker/novt_resize/test_side_00000_resize.xml"
    image_path = "/mnt/huanyuan2/data/image/HY_Tanker/novt_resize/7118000000000000-221116-225400-225420-01p013000000_00000_resize.xml"

    caffe_model = "/home/huanyuan/share/huanyuan/GAF/huanyuan/novt/Tanker/Tanker/tanker_1125.caffemodel"
    prototxt_file = "/home/huanyuan/share/huanyuan/GAF/huanyuan/novt/Tanker/Tanker/deploy.prototxt"
    size = (256, 144)

    # caffe.set_device(0)
    # caffe.set_mode_gpu()
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_file, caffe_model, caffe.TEST)
    net.blobs['data'].reshape(1, 3, size[1], size[0])   # N C H W

    # 加载矩阵
    cv_file = cv2.FileStorage(image_path, cv2.FILE_STORAGE_READ)
    img_origin = cv_file.getNode("test").mat()
    cv_file.release()

    img_origin = cv2.resize(img_origin, size)
    img = img_origin.astype(np.float32)
    img = img.transpose((2, 0, 1))
    net.blobs['data'].data[...] = img
    
    output = net.forward()['decon6_out'][0][0]
    output = output * 255
    print(output)
    cv_file = cv2.FileStorage(os.path.join("/mnt/huanyuan2/data/image/HY_Tanker/pc_load_novt/", "caffe_res_{}_xml.xml".format(os.path.basename(image_path)[:-4])), cv2.FILE_STORAGE_WRITE)
    cv_file.write('test', output)
    cv_file.release()

    img_origin[:,:,2][np.where(output > 100)] = 255
    contours, hierarchy = cv2.findContours(output.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    output_path = os.path.join("/mnt/huanyuan2/data/image/HY_Tanker/pc_load_novt/", "caffe_res_{}_xml.jpg".format(os.path.basename(image_path)[:-4]))
    cv2.imwrite(output_path, img_origin)


def load_novt_jpg_test():
    # image_path = "/mnt/huanyuan2/data/image/HY_Tanker/novt_resize/test_side_00000_resize.xml"
    image_path = "/mnt/huanyuan2/data/image/HY_Tanker/novt_resize/7118000000000000-221116-225400-225420-01p013000000_00000_resize.jpg"

    caffe_model = "/home/huanyuan/share/huanyuan/GAF/huanyuan/novt/Tanker/Tanker/tanker_1125.caffemodel"
    prototxt_file = "/home/huanyuan/share/huanyuan/GAF/huanyuan/novt/Tanker/Tanker/deploy.prototxt"
    size = (256, 144)

    # caffe.set_device(0)
    # caffe.set_mode_gpu()
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_file, caffe_model, caffe.TEST)
    net.blobs['data'].reshape(1, 3, size[1], size[0])   # N C H W

    # 加载矩阵
    img_origin = cv2.imread(image_path)

    img = img_origin.astype(np.float32)
    img = img.transpose((2, 0, 1))
    net.blobs['data'].data[...] = img
    
    output = net.forward()['decon6_out'][0][0]
    output = output * 255
    print(output)
    cv_file = cv2.FileStorage(os.path.join("/mnt/huanyuan2/data/image/HY_Tanker/pc_load_novt/", "caffe_res_{}_xml.xml".format(os.path.basename(image_path)[:-4])), cv2.FILE_STORAGE_WRITE)
    cv_file.write('test', output)
    cv_file.release()

    img_origin[:,:,2][np.where(output > 100)] = 255
    contours, hierarchy = cv2.findContours(output.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    output_path = os.path.join("/mnt/huanyuan2/data/image/HY_Tanker/pc_load_novt/", "caffe_res_{}_xml.jpg".format(os.path.basename(image_path)[:-4]))
    cv2.imwrite(output_path, img_origin)


if __name__ == "__main__":
    test()

    # load_novt_jpg_test()
    # load_novt_xml_test()
