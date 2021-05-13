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

# xiaoan: tc_resnet14_amba
caffe_prototxt = "/mnt/huanyuan/model/audio_model/novt_model/kws_xiaoan8k_tc_resnet14/kws_xiaoan8k_tc_resnet14_2_2_05112021.prototxt"
caffe_model = "/mnt/huanyuan/model/audio_model/novt_model/kws_xiaoan8k_tc_resnet14/kws_xiaoan8k_tc_resnet14_2_2_05112021.caffemodel"
# caffe_prototxt = "/mnt/huanyuan/model/audio_model/novt_model/kws_xiaoan8k_tc_resnet14/kws_xiaoan8k_tc_resnet14_2_2_04162021.prototxt"
# caffe_model = "/mnt/huanyuan/model/audio_model/novt_model/kws_xiaoan8k_tc_resnet14/kws_xiaoan8k_tc_resnet14_2_2_04162021.caffemodel"
model_output = "Softmax"
image_transpose = [0, 1, 3, 2]
model_input_size = [1, 1, 48, 146]

net = caffe.Net(caffe_prototxt, caffe_model, caffe.TEST)
image_size = [1, 1, 146, 48]

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
    img = cv2.imread("/home/huanyuan/share/audio_data/weakup_xiaoan8k/image_48_146_temp/RM_KWS_XIAOAN_xiaoan_S009M1D11T5_3200.jpg", 0)

    # img = np.ones(image_size, dtype=np.float32)
    # img = np.transpose(img, axes=image_transpose)

    time_caffe, caffe_blobs, caffe_params = forward_caffe(caffe_prototxt, caffe_model, img)

    print('caffe forward time %d', time_caffe)

    print('------------ Output Difference ------------')
    net_output = caffe_blobs[model_output].data[0]
    print(net_output.shape)
    print(net_output)
