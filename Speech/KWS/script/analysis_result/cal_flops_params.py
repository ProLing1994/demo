import importlib
import torch
import sys

from thop import profile   
from thop import clever_format
sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')


def main():
    batch_size = 1
    in_channels = 1
    image_height = 196
    image_weidth = 64
    num_classes = 2

    # net_name = "crnn-attention"
    # net_name = "crnn-avg"
    net_name = "res15"
    # net_name = "res15-narrow"
    # net_name = "res15-narrow-amba"
    # net_name = "wavenet"
    # net_name = "edge-speech-nets"
    # net_name = "tc-resnet8"
    # net_name = "tc-resnet8-dropout"
    # net_name = "tc-resnet14"
    # net_name = "tc-resnet14-dropout"
    # net_name = "tc-resnet18-dropout"
    # net_name = "tc-resnet14-amba"

    # load network structure
    net_module = importlib.import_module('network.' + net_name)
    net = net_module.SpeechResModel(num_classes=num_classes, 
                                    image_height=image_height, 
                                    image_weidth=image_weidth)

    input = torch.randn(batch_size, in_channels, image_height, image_weidth)
    flops, params = profile(net, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print("par.: ", params)
    print("Mult.:", flops)

if __name__ == "__main__":
    main()