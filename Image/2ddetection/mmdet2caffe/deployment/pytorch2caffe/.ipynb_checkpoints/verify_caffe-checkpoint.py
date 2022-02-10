import time
import torch
import argparse
import numpy as np
import mmcv

import sys
sys.path.append('./')

from mmdet.apis import init_detector

def load_data(input_img_size):
    size = (1, 3) + input_img_size
    data = np.random.rand(*size)
    return data


def forward_pytorch(pt_path, input, cfg, device=torch.device('cpu')):
    model = init_detector(cfg, pt_path, device)
    input = torch.from_numpy(input)
    input = input.to(torch.float32)
    t0 = time.time()
    blobs = model.forward_caffe(input)
    t1 = time.time()
    return t1 - t0, blobs, model.parameters()


def forward_caffe(protofile, weightfile, input):
    caffe_root = "/phzhang/caffe-master/"
    sys.path.insert(0, caffe_root + "python")
    import caffe

    caffe.set_mode_cpu()
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    # net.blobs['data'].reshape(1, 3, height, width)
    net.blobs["blob1"].data[...] = input
    t0 = time.time()
    output = net.forward()
    t1 = time.time()
    return t1 - t0, net.blobs, net.params


def compare():
    model = args.pytorch_model
    protofile = args.caffe_protofile
    weightfile = args.caffe_weightfile
    caffe_blob_name = args.caffe_blob_name

    config = mmcv.Config.fromfile(args.config_file)

    for op in config.data.test.pipeline:
        if op['type'] == "MultiScaleFlipAug":
            scale = op['img_scale']
            break

    input = load_data(scale)

    time_pytorch, pytorch_blobs, _ = forward_pytorch(model, input, config)
    time_caffe, caffe_blobs, _ = forward_caffe(protofile, weightfile, input)

    print("pytorch forward time ", time_pytorch)
    print("caffe forward time ", time_caffe)

    pytorch_data = pytorch_blobs[0][1][0].data.numpy().flatten()

    caffe_data = caffe_blobs[caffe_blob_name].data[0][...].flatten()
    print(max(pytorch_data))

    print(max(caffe_data))
    # pytorch_data = pytorch_blobs.data.numpy().flatten()
    # caffe_data = caffe_blobs[caffe_blob_name].data[0][...].flatten()
    print("-----------------torch data------------------")
    print(pytorch_data)
    print("-----------------caffe data------------------")
    print(caffe_data)
    print("----------------average diff-----------------")
    diff = abs(pytorch_data - caffe_data).sum()
    print(
        "%s\npytorch_shape: %-s\ncaffe_shape: %s\noutput_diff: %f"
        % (
            caffe_blob_name,
            pytorch_data.shape,
            caffe_data.shape,
            diff / pytorch_data.size,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--caffe_protofile",
        default="/phzhang/mmdetection-master-zhiguan/caffe_model/yoloxv2_guoshengdaoData_patch_mix.prototxt",
        type=str,
    )
    parser.add_argument(
        "--caffe_weightfile",
        default="/phzhang/mmdetection-master-zhiguan/caffe_model/yoloxv2_guoshengdaoData_patch_mix.caffemodel",
        type=str,
    )
    parser.add_argument(
        "--config_file",
        default="/phzhang/mmdetection-master-zhiguan/work_dirs/yoloxv2_guoshengdaoData_patch_mix/yoloxv2_rmvgg_rmfpn_nova_3level_zhiguan_non_car_person_detection_320x320.py",
        type=str,
    )
    parser.add_argument(
        "--pytorch_model", default="/phzhang/mmdetection-master-zhiguan/work_dirs/yoloxv2_guoshengdaoData_patch_mix/epoch_24.pth", type=str
    )
    parser.add_argument("--caffe_blob_name", type=str, default="blob1")
    args = parser.parse_args()
    compare()
