import cv2
import torch
import argparse
import numpy as np

import sys
sys.path.append('./')



def load_data(image, input_img_size):
    # size = (1, 3) + input_img_size
    # data = np.random.rand(*size)
    data = cv2.resize(image, input_img_size).astype(np.float32)
    data -= np.array([127.5, 127.5, 127.5], dtype=np.float32)
    data /= 57.375
    data = data[..., ::-1]
    data = np.transpose(data, (2, 0, 1))
    data = data[np.newaxis, :, :, :]
    return data


def forward_caffe(protofile, weightfile, input):
    caffe_root = "/phzhang/caffe-master/"
    sys.path.insert(0, caffe_root + "python")
    import caffe

    caffe.set_mode_cpu()
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    # net.blobs['data'].reshape(1, 3, height, width)
    net.blobs["blob1"].data[...] = input
    output = net.forward()
    return net.blobs, net.params


def forward():
    protofile = args.caffe_protofile
    weightfile = args.caffe_weightfile
    scale = (args.scale, args.scale)

    img = cv2.imread("./test.jpg")
    image_h, image_w, _ = img.shape
    input = load_data(img, scale)

    caffe_blobs, _ = forward_caffe(protofile, weightfile, input)

    cls_preds = []
    bbox_preds = []

    cls_preds.append(caffe_blobs["mul_blob1"].data[0][...].flatten())
    bbox_preds.append(caffe_blobs["relu_blob49"].data[0][...].flatten())

    cls_preds.append(caffe_blobs["mul_blob2"].data[0][...].flatten())
    bbox_preds.append(caffe_blobs["relu_blob50"].data[0][...].flatten())

    cls_preds.append(caffe_blobs["mul_blob3"].data[0][...].flatten())
    bbox_preds.append(caffe_blobs["relu_blob51"].data[0][...].flatten())

    featuremap_sizes = [40, 20, 10]

    for cls_pred, bbox_pred, featuremap_size in zip(cls_preds, bbox_preds, featuremap_sizes):
        cls_pred = torch.Tensor(cls_pred)
        bbox_pred = torch.Tensor(bbox_pred).reshape(4, featuremap_size * featuremap_size).permute(1, 0)
        _, topk_inds = torch.topk(cls_pred, 100)

        scores = cls_pred[topk_inds]
        inds = scores > 0.3
        topk_inds = topk_inds[inds]
        bbox_pred = bbox_pred[topk_inds]

        for pos, ltrb in zip(topk_inds, bbox_pred):
            h, w = pos // featuremap_size, pos % featuremap_size
            l, t, r, b = ltrb
            # l = w + 0.5 - l
            # t = h + 0.5 - t
            # r = w + 0.5 + r
            # b = h + 0.5 + b

            # l = l * image_w / 40
            # t = t * image_h / 40
            # r = r * image_w / 40
            # b = b * image_h / 40
            l = w + 0.5 - l
            t = h + 0.5 - t
            r = w + 0.5 + r
            b = h + 0.5 + b

            l = l * image_w / featuremap_size
            t = t * image_h / featuremap_size
            r = r * image_w / featuremap_size
            b = b * image_h / featuremap_size

            img = cv2.rectangle(img, (l, t), (r, b), (255, 0, 0))
        cv2.imwrite("./draw_test.jpg", img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--caffe-protofile",
        default="/phzhang/mmdetection-master-zhiguan/caffe_model/yoloxv2_guoshengdaoData_patch_mix.prototxt",
        type=str,
    )
    parser.add_argument(
        "--caffe-weightfile",
        default="/phzhang/mmdetection-master-zhiguan/caffe_model/yoloxv2_guoshengdaoData_patch_mix.caffemodel",
        type=str,
    )
    parser.add_argument("--scale", type=int, default=320)
    args = parser.parse_args()
    forward()
