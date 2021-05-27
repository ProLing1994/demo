import cv2
import numpy as np
import os
import sys
import time

caffe_root = "/home/huanyuan/code/caffe_ssd/"
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')
import caffe

# # asr: english 
# # caffe_prototxt = "/home/huanyuan/share/hisi/KWS_model/kws/models/asr_english_16k_0202.prototxt"
# caffe_prototxt = "/home/huanyuan/share/hisi/KWS_model/kws/models/asr_english_16k_softmax_0202.prototxt"
# caffe_model = "/home/huanyuan/share/hisi/KWS_model/kws/models/asr_english_16k_0202.caffemodel"
# model_output = "prob"
# image_transpose = [0, 1, 2, 3]
# model_input_size = [1, 1, 296, 64]
# image_size = [1, 1, 296, 64]

# # asr: mandarin taxi 
# caffe_prototxt = "/home/huanyuan/share/hisi/KWS_model/kws/models/asr_mandarin_taxi_16k_64dim.prototxt"
# caffe_model = "/home/huanyuan/share/hisi/KWS_model/kws/models/asr_mandarin_taxi_16k_64dim.caffemodel"
# model_output = "prob"
# image_transpose = [0, 1, 2, 3]
# model_input_size = [1, 1, 396, 64]
# image_size = [1, 1, 396, 64]

# xiaoan8k: tc_resnet14_amba
caffe_prototxt = "/mnt/huanyuan/model/audio_model/hisi_model/kws_xiaoan8k_tc_resnet14/kws_xiaoan8k_tc_resnet14_hisi_3_1_05272021.prototxt"
caffe_model = "/mnt/huanyuan/model/audio_model/hisi_model/kws_xiaoan8k_tc_resnet14/kws_xiaoan8k_tc_resnet14_hisi_3_1_05272021.caffemodel"
model_output = "prob"
image_transpose = [0, 1, 3, 2]
model_input_size = [1, 1, 48, 144]
image_size = [1, 1, 144, 48]

net = caffe.Net(caffe_prototxt, caffe_model, caffe.TEST)

def forward_caffe(protofile, weightfile, image):

    # caffe.set_device(0)
    # caffe.set_mode_gpu()
    caffe.set_mode_cpu()

    net = caffe.Net(protofile, weightfile, caffe.TEST)
    net.blobs['data'].reshape(*model_input_size)
    net.blobs['data'].data[...] = image
    t0 = time.time()
    output = net.forward()
    print(output)
    t1 = time.time()
    return t1-t0, net.blobs, net.params


if __name__ == '__main__':
    # img = np.ones(image_size, dtype=np.float32)
    # if image_transpose != [0, 1, 2, 3]:
    #     img = np.transpose(img, axes=image_transpose)
    img = cv2.imread("/home/huanyuan/share/audio_data/weakup_xiaoan8k/image_48_144/RM_KWS_XIAOAN_xiaoan_S021M1D10T12_0.jpg", 0)

    time_caffe, caffe_blobs, caffe_params = forward_caffe(caffe_prototxt, caffe_model, img)

    print('caffe forward time %d', time_caffe)

    print('------------ Output Difference ------------')
    net_output = caffe_blobs[model_output].data[0]
    print(net_output.shape)
    print(net_output)
