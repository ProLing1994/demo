import cv2
import math
import numpy as np 
import os
import sys
import time
caffe_root = '/home/huanyuan/code/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe

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
	net_output = net.forward()['detection_out']
	# net_output = net.forward()['DetectionOutput']

	box = net_output[0,0,:,3:7] * np.array([width, height, width, height])
	box = box.astype(np.int32)

	cls = net_output[0,0,:,1]
	conf = net_output[0,0,:,2]

	for i in range(len(box)):
		if conf[i] > conf_thresh:
			p1 = (box[i][0], box[i][1])
			p2 = (box[i][2], box[i][3])
			cv2.rectangle(img, p1, p2, (0,0,255))
			p3 = (max(p1[0], 15), max(p1[1], 15))
			title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
			cv2.putText(img, title, p3, cv2.FONT_ITALIC, 0.6, (0, 0, 255), 1)
	
	output_path = os.path.join(output_image_dir, os.path.basename(image_path))
	cv2.imwrite(output_path, img)
	return 
		
if __name__ == "__main__":
	input_image_dir = "/mnt/huanyuan/model_final/image_model/ssd_rfb_jct_zg/image_2M_5M/"
	output_image_dir = "/mnt/huanyuan/model_final/image_model/ssd_rfb_jct_zg/test_result"

	# ssd(default)
	# ssd_model = "/home/huanyuan/code/models/mobilenet_iter_73000.caffemodel"
	# ssd_prototxt = "/home/huanyuan/code/models/mobilenet_iter_73000.deploy.prototxt"
	# ssd_param_CHW  = [3, 300, 300]
	# ssd_mean  = [127.5, 127.5, 127.5]
	# ssd_scale  = 0.007843
	# CLASSES = ('background',
	#             'aeroplane', 'bicycle', 'bird', 'boat',
	#             'bottle', 'bus', 'car', 'cat', 'chair',
	#             'cow', 'diningtable', 'dog', 'horse',
	#             'motorbike', 'person', 'pottedplant',
	#             'sheep', 'sofa', 'train', 'tvmonitor')

	# # face
	# ssd_model = "/home/huanyuan/code/models/ssd_face_mask.caffemodel"
	# ssd_prototxt = "/home/huanyuan/code/models/ssd_face_mask.prototxt"
	# ssd_param_CHW  = [3, 300, 300]
	# ssd_mean  = [127.5, 127.5, 127.5]
	# ssd_scale  = 0.007843
	# CLASSES = ('background', "face")

	# car
	# ssd_model = "/home/huanyuan/code/models/ssd_car_0710.caffemodel"
	# ssd_prototxt = "/home/huanyuan/code/models/ssd_car_0710.prototxt"
	# ssd_param_CHW  = [3, 300, 300]
	# ssd_mean  = [104.0, 117.0, 123.0]
	# conf_thresh = 0.5
	# ssd_scale  = 1.0
	# CLASSES = ('background', "car")

	# License_plate
	# ssd_model = "/home/huanyuan/code/models/ssd_License_plate_mobilenetv2.caffemodel"
	# ssd_prototxt = "/home/huanyuan/code/models/ssd_License_plate_mobilenetv2_fpn_ncnn_concat.prototxt"
	# ssd_param_CHW  = [3, 300, 300]
	# ssd_mean  = [104.0, 117.0, 123.0]
	# ssd_scale  = 1.0
	# CLASSES = ('background', "License_plate")
	
	# car_bus_truck_non_motorized_person
	ssd_model = "/mnt/huanyuan/model_final/image_model/ssd_rfb_jct_zg/car_bus_truck_non_motorized_person_zg_2022-06-27-21/SSD_VGG_FPN_RFB_VOC_car_bus_truck_non_motorized_person_zg_2022-06-27-21.caffemodel"
	ssd_prototxt = "/mnt/huanyuan/model_final/image_model/ssd_rfb_jct_zg/car_bus_truck_non_motorized_person_zg_2022-06-27-21/FPN_RFB_4class_noDilation_prior.prototxt"
	ssd_param_CHW  = [3, 300, 300]
	ssd_mean  = [104, 117, 123]
	ssd_scale  = 1.0
	CLASSES = ('background', 'car_bus_truck', 'non_motorized', 'person')

	# something wronge
	# ssd_model = "/home/huanyuan/code/MNN/models/face.caffemodel"
	# ssd_prototxt = "/home/huanyuan/code/MNN/models/face.prototxt"
	# ssd_param_CHW  = [3, 300, 300]
	# ssd_mean  = [104.0, 117.0, 123.0]
	# ssd_scale  = 1.0
	# CLASSES = ('background', "face", "phone")

	ssd_param = [ssd_mean, ssd_scale]
	ssd_net = ssd_init(ssd_prototxt, ssd_model, ssd_param_CHW)

	start = time.clock()
	for image_name in os.listdir(input_image_dir):
		image_path = os.path.join(input_image_dir, image_name)
		ssd_detect(ssd_net, image_path, ssd_param, ssd_param_CHW, conf_thresh=0.5)
	end = time.clock()
	print("average time= {}s".format((end - start)/len(os.listdir(input_image_dir))))
