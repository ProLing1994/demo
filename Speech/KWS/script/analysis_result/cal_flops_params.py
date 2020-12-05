import importlib
import torch
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from thop import profile   
from thop import clever_format

def main():
    # net_name = "crnn-attention"
    # net_name = "crnn-avg"
    net_name = "res15"

    # load network structure
    net_module = importlib.import_module('network.' + net_name)

    net = net_module.SpeechResModel(num_classes=3, 
                                    image_height=101, 
                                    image_weidth=40)

    input = torch.randn(1, 1, 101, 40)
    flops, params = profile(net, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print("par.: ", params)
    print("Mult.:", flops)

if __name__ == "__main__":
    main()