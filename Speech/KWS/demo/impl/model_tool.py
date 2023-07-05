import importlib
import numpy as np
import os
import sys

# caffe_root = "/home/huanyuan/code/caffe/"
caffe_root = '/home/huanyuan/code/caffe_ssd-ssd-gpu/'
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')
import caffe
import torch

def caffe_model_init(prototxt, model, net_input_name, CHW_params, use_gpu=False):
    if use_gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
        print("[Information:] GPU mode")
    else:
        caffe.set_mode_cpu()
        print("[Information:] CPU mode")
    net = caffe.Net(prototxt, model, caffe.TEST)
    net.blobs[net_input_name].reshape(1, int(CHW_params[0]), int(CHW_params[1]), int(CHW_params[2])) 
    return net


def caffe_model_forward(net, feature_data, input_name, output_name, bool_kws_transpose=False):
    if bool_kws_transpose:
        feature_data = feature_data.T

    # net.blobs[cfg.model.kws_net_input_name].data[...] = np.expand_dims(feature_data, axis=0)
    net.blobs[input_name].data[...] = np.expand_dims(np.expand_dims(feature_data, axis=0), axis=0)
    net_output = net.forward()[output_name]
    return net_output


def pytorch_kws_model_init(chk_file, model_name, class_name, num_classes, image_height, image_weidth, use_gpu=False):
    # init model
    net_module = importlib.import_module('network.' + model_name)
    net = net_module.__getattribute__(class_name)(num_classes=num_classes,
                                                    image_height=image_height,
                                                    image_weidth=image_weidth)
    
    if use_gpu:
        net = net.cuda()

    # load state
    state = torch.load(chk_file)
    new_state = {}
    for k,v in state['state_dict'].items():
        name = k[7:]
        new_state[name] = v
    net.load_state_dict(new_state)

    net.eval()
    return net


def pytorch_model_forward(net, feature_data, use_gpu=False):
    data_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(feature_data, axis=0), axis=0))
    data_tensor = data_tensor.float()

    if use_gpu:
        data_tensor = data_tensor.cuda()
    net_output = net(data_tensor).cpu().data.numpy()
    return net_output


def pytorch_asr_model_init(chk_file, model_name, class_name, num_classes, use_gpu=False):
    # init model 
    net_module = importlib.import_module('network.' + model_name)
    net = net_module.__getattribute__(class_name)(num_classes)

    if use_gpu:
        net = net.cuda()

    # load state
    checkpoint=torch.load(os.path.join(chk_file))
    net.load_state_dict(checkpoint['state_dict'], strict=True)

    net.eval()
    return net