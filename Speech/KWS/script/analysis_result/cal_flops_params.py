import importlib
import torch
import sys

from thop import profile   
from thop import clever_format

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.utils.train_tools import *

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    batch_size = 1
    in_channels = 1
    # image_height = 196
    # image_weidth = 48
    image_height = 101
    image_weidth = 40

    # net_name = "crnn-attention"
    # net_name = "crnn-avg"
    # net_name = "res15"
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
    # net_name = "tc-resnet14-hisi"
    # net_name = "tc-resnet14-amba-novt-196"
    net_name = "bc-resnet"

    # load configuration file
    config_file = "/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_embedding_xiaoan8k.py"
    cfg = load_cfg_file(config_file)

    # load network structure
    net_module = importlib.import_module('network.' + net_name)
    # net = net_module.SpeechResModel(cfg)
    net = net_module.BCResNet(cfg)

    input = torch.randn(batch_size, in_channels, image_height, image_weidth)
    flops, params = profile(net, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print("par.: ", params)
    print("Mult.: ", flops)

    params_num = count_parameters(net)
    print("par.:  {}K".format(params_num/1000.0))

if __name__ == "__main__":
    main()