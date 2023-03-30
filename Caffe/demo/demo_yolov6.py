import cv2
import math
import numpy as np 
import os
import sys
import time

# caffe_root = '/home/huanyuan/code/caffe_ssd-ssd/'
caffe_root = '/home/huanyuan/code/caffe_ssd-ssd-gpu/'
sys.path.insert(0, caffe_root+'python')
import caffe

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def ssd_init(ssd_prototxt, ssd_model, ssd_param_CHW):
	caffe.set_device(0)
	caffe.set_mode_gpu()
	net = caffe.Net(ssd_prototxt, ssd_model, caffe.TEST)
	net.blobs['data'].reshape(1, 3, ssd_param_CHW[1], ssd_param_CHW[2]) 
	return net

def pre_process(img, ssd_param, ssd_param_CHW):
	img = cv2.resize(img, (ssd_param_CHW[2], ssd_param_CHW[1]))
	img = img.astype(np.float32)
	img -= ssd_param[0]
	img = img * ssd_param[1]
	img = img.transpose((2, 0, 1))
	return img 

def ssd_detect(net, image_path, ssd_param, ssd_param_CHW, conf_thresh=0.5):
    print("ssd detect image: {}".format(image_path))
    img = cv2.imread(image_path)
    height = img.shape[0]
    width = img.shape[1]

    net.blobs['data'].data[...] = pre_process(img, ssd_param, ssd_param_CHW)
    conv44 = net.forward()['conv_blob44']
    conv45 = net.forward()['conv_blob45']
    conv46 = net.forward()['conv_blob46']

    conv49 = net.forward()['conv_blob49']
    conv50 = net.forward()['conv_blob50']
    conv51 = net.forward()['conv_blob51']
    
    conv54 = net.forward()['conv_blob54']
    conv55 = net.forward()['conv_blob55']
    conv56 = net.forward()['conv_blob56']
    
    for idx in range(conv44.shape[2]):
	    for idy in range(conv44.shape[3]):
		    conv44[0, 0, idx, idy] = sigmoid(conv44[0, 0, idx, idy])
    for idx in range(conv49.shape[2]):
	    for idy in range(conv49.shape[3]):
		    conv49[0, 0, idx, idy] = sigmoid(conv49[0, 0, idx, idy])
    for idx in range(conv54.shape[2]):
	    for idy in range(conv54.shape[3]):
		    conv54[0, 0, idx, idy] = sigmoid(conv54[0, 0, idx, idy])

    print(conv54)
    return 
		
if __name__ == "__main__":
	input_image_dir = "/home/huanyuan/share/huanyuan/Brazil_ANPR_5M_NOVT_ST/huanyuan/image_data/C28_data/test/"
	output_image_dir = "/home/huanyuan/share/huanyuan/Brazil_ANPR_5M_NOVT_ST/huanyuan/image_data/C28_data/"

	ssd_model = "/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0327/yolov6_rm_c28.caffemodel"
	ssd_prototxt = "/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0327/yolov6_rm_c28.prototxt"
	ssd_param_CHW  = [3, 288, 512]
	ssd_mean  = [0, 0, 0]
	ssd_scale  = 1.0
	conf_thresh = 0.25
	CLASSES = ('background', 'License_plate')

	ssd_param = [ssd_mean, ssd_scale]
	ssd_net = ssd_init(ssd_prototxt, ssd_model, ssd_param_CHW)

	start = time.clock()
	img_list = os.listdir(input_image_dir)
	img_list.sort()
	for image_name in img_list:
		image_path = os.path.join(input_image_dir, image_name)
		ssd_detect(ssd_net, image_path, ssd_param, ssd_param_CHW, conf_thresh=conf_thresh)
	end = time.clock()
	print("average time= {}s".format((end - start)/len(img_list)))
