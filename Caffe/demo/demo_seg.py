import cv2
import numpy as np 
import os
import sys
import time

caffe_root = '/home/huanyuan/code/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe


if __name__ == "__main__":

    image_path = "/home/huanyuan/code/third_code/image/tanker/Seg/video/jpg/pic_00032.jpg"
    caffe_model = "/home/huanyuan/code/third_code/image/tanker/Seg/tanker_0905_B.caffemodel"
    prototxt_file = "/home/huanyuan/code/third_code/image/tanker/Seg/caffe_tanker_B.prototxt"
    size = (256, 144)

    # caffe.set_device(0)
    # caffe.set_mode_gpu()
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_file, caffe_model, caffe.TEST)
    net.blobs['data'].reshape(1, 3, size[1], size[0])   # N C H W

    img_origin = cv2.imread(image_path)

    img_origin = cv2.resize(img_origin, size)
    img = img_origin.astype(np.float32)
    img = img.transpose((2, 0, 1))
    net.blobs['data'].data[...] = img
    
    output = net.forward()['decon6_out'][0][0]
    output = output * 255
    
    img_origin[:,:,2][np.where(output > 100)] = 255
    contours, hierarchy = cv2.findContours(output.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    output_path = os.path.join("/home/huanyuan/code/third_code/image/tanker/Seg/video/", "caffe_res_{}.jpg".format(os.path.basename(image_path)))
    cv2.imwrite(output_path, img_origin)