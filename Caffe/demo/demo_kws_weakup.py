import cv2
import numpy as np
import os
import sys
import time

caffe_root = "/home/huanyuan/code/caffe_ssd/"
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')
import caffe

# res15_narrow_amba
# caffe_prototxt = "/mnt/huanyuan/model/audio_model/kws_xiaorui_res15_narrow/res15_narrow_amba_03112011.prototxt"
# caffe_model = "/mnt/huanyuan/model/audio_model/kws_xiaorui_res15_narrow/res15_narrow_amba_03112011.caffemodel"
# model_output = "fc_blob1"
# image_transpose = [0, 1, 2, 3]
# model_input_size = [1, 1, 201, 40]

# tc_resnet14_amba
caffe_prototxt = "/mnt/huanyuan/model/audio_model/kws_xiaorui_tc_resnet14/tc_resnet14_amba_031120221.prototxt"
caffe_model = "/mnt/huanyuan/model/audio_model/kws_xiaorui_tc_resnet14/tc_resnet14_amba_031120221.caffemodel"
model_output = "conv_blob23"
image_transpose = [0, 1, 3, 2]
model_input_size = [1, 1, 40, 201]

net = caffe.Net(caffe_prototxt, caffe_model, caffe.TEST)
image_size = [1, 1, 201, 40]

def forward_caffe(protofile, weightfile, image):

    # caffe.set_device(0)
    # caffe.set_mode_gpu()
    caffe.set_mode_cpu()

    net = caffe.Net(protofile, weightfile, caffe.TEST)
    net.blobs['data'].reshape(*model_input_size)
    net.blobs['data'].data[...] = image
    t0 = time.time()
    output = net.forward()
    t1 = time.time()
    return t1-t0, net.blobs, net.params


if __name__ == '__main__':
    img = np.ones(image_size, dtype=np.float32)
    img = np.transpose(img, axes=image_transpose)

    time_caffe, caffe_blobs, caffe_params = forward_caffe(caffe_prototxt, caffe_model, img)

    print('caffe forward time %d', time_caffe)

    print('------------ Output Difference ------------')
    net_output = caffe_blobs[model_output].data[0]
    print(net_output.shape)
    print(net_output)
