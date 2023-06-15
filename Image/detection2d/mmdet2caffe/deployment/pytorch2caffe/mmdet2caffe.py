from argparse import ArgumentParser

import torch
import mmcv

import sys
sys.path.append("./")

from mmdet.apis import init_detector


def get_parser():
    parser = ArgumentParser(description="Convert Pytorch to Caffe model")

    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
        default="",
    )
    parser.add_argument(
        "--model-path",
        metavar="FILE",
        help="path to model",
        default="",
    )
    parser.add_argument("--name", default="", help="name for converted model")
    parser.add_argument(
        "--output",
        default="",
        help="path to save converted onnx model",
    )
    return parser


if __name__ == "__main__":
    import pytorch_to_caffe

    args = get_parser().parse_args()

    # args.config_file = "/mnt/huanyuan/model/image/yolov6/yolox_landmark_wider_face_20230609/yolovx_face_wider_face.py"
    # args.model_path = "/mnt/huanyuan/model/image/yolov6/yolox_landmark_wider_face_20230609/epoch_300.pth"
    # args.name = "yolox_landmark_wider_face_20230609"
    # args.output = "/mnt/huanyuan/model/image/yolov6/yolox_landmark_wider_face_20230609"

    config = mmcv.Config.fromfile(args.config_file)

    # build the model from a config file and a checkpoint file
    model = init_detector(config, args.model_path, device='cpu')

    for op in config.data.test.pipeline:
        if op['type'] == "MultiScaleFlipAug":
            scale = op['img_scale']
            break

    size = (1, 3) + (scale[1], scale[0])
    inputs = torch.randn(size)

    mmcv.mkdir_or_exist(args.output)
    pytorch_to_caffe.trans_net(model, inputs, args.name)
    pytorch_to_caffe.save_prototxt(f"{args.output}/{args.name}.prototxt")
    pytorch_to_caffe.save_caffemodel(f"{args.output}/{args.name}.caffemodel")

    print(f"Export caffe model in {args.output} successfully!")
