import cv2
import numpy as np
import os
import sys
import time

caffe_root = "/home/huanyuan/code/caffe/"
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')
import caffe


net_file = "/mnt/huanyuan/model/audio_model/kws_xiaorui_res15_narrow/res15_narrow_amba_03112011.prototxt"
caffe_model = "/mnt/huanyuan/model/audio_model/kws_xiaorui_res15_narrow/res15_narrow_amba_03112011.caffemodel"

net = caffe.Net(net_file, caffe_model, caffe.TEST)
size = (201, 40)


def forward_caffe(protofile, weightfile, image):

    # caffe.set_device(0)
    # caffe.set_mode_gpu()
    caffe.set_mode_cpu()

    net = caffe.Net(protofile, weightfile, caffe.TEST)
    net.blobs['data'].reshape(1, 1, size[0], size[1])
    net.blobs['data'].data[...] = image
    t0 = time.time()
    output = net.forward()
    t1 = time.time()
    return t1-t0, net.blobs, net.params


if __name__ == '__main__':
    img = np.ones([1, 1, size[0], size[1]], dtype=np.float32)

    time_caffe, caffe_blobs, caffe_params = forward_caffe(net_file, caffe_model, img)

    print('caffe forward time %d', time_caffe)

    print('------------ Output Difference ------------')
    # net_output = caffe_blobs["add_blob6"].data[0]
    # net_output = caffe_blobs["relu_blob14"].data[0]
    # net_output = caffe_blobs["batch_norm_blob13"].data[0]
    net_output = caffe_blobs["fc_blob1"].data[0]
    print(net_output.shape)
    print(net_output)
