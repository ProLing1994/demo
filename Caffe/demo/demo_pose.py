# !/user/bin/env python
# coding=utf-8
"""
@project : mmpose-master
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : caffe_forward.py
#@time   : 2021-03-03 20:53:20
"""
import cv2
import numpy as np
import sys
import time

caffe_root = "/home/huanyuan/code/caffe/"
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')
import caffe

net_name = 'vgg'
if net_name == 'mobilenet':
    net_file = 'depoly/caffemodels/mobilenetv2/mobilenetv2.prototxt'
    caffe_model = 'depoly/caffemodels/mobilenetv2/mobilenetv2.caffemodel'
elif net_name == 'vgg':
    net_file = "/home/huanyuan/share/novt/POSE_model/Convolution1.prototxt"
    # net_file = "/home/huanyuan/share/novt/POSE_model/softmax.prototxt"
    # net_file = "/home/huanyuan/share/novt/POSE_model/deploy.prototxt"
    caffe_model = "/home/huanyuan/share/novt/POSE_model/deploy.caffemodel"
else:
    raise ValueError('we are not support %s yet' % net_name)

img_path = "/home/huanyuan/share/pose_data/00000.jpg"

net = caffe.Net(net_file, caffe_model, caffe.TEST)
size = (256, 256)


def forward_caffe(protofile, weightfile, image):

    # caffe.set_device(0)
    # caffe.set_mode_gpu()
    caffe.set_mode_cpu()

    net = caffe.Net(protofile, weightfile, caffe.TEST)
    net.blobs['data'].reshape(1, 3, size[1], size[0])
    net.blobs['data'].data[...] = image
    t0 = time.time()
    output = net.forward()
    t1 = time.time()
    return t1-t0, net.blobs, net.params


if __name__ == '__main__':
    print('This is main ....')

    # img = np.ones([1, 3, size[1], size[0]], dtype=np.float32)
    img = cv2.imread(img_path)
    img = (img - 127.5)/255/0.225
    img = cv2.resize(img, size)
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, :]

    time_caffe, caffe_blobs, caffe_params = forward_caffe(net_file, caffe_model, img)

    print('caffe forward time %d', time_caffe)

    print('------------ Output Difference ------------')
    net_output = caffe_blobs["Convolution1"].data[0]
    # net_output = caffe_blobs["Softmax"].data[0] 
    print(net_output.shape)
    print(net_output)

    # heatmap_blob_name = "Sigmoid1"
    # tag_blob_name = "Convolution18"

    # # det_caffe_data = caffe_blobs[heatmap_blob_name].data[0][...].flatten()
    # # box_caffe_data = caffe_blobs[tag_blob_name].data[0][...].flatten()
    # det_caffe_data = caffe_blobs[heatmap_blob_name].data[0][...]
    # box_caffe_data = caffe_blobs[tag_blob_name].data[0][...]
    # print(det_caffe_data)
    # print(box_caffe_data)
