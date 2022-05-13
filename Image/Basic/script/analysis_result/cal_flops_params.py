import importlib
import torch
import sys
import time 
from tqdm import tqdm

from thop import profile   
from thop import clever_format

sys.path.insert(0, '/home/huanyuan/code/demo/Image/detection2d/ssd_rfb_crossdatatraining/')
from models.SSD_VGG_Optim_FPN_RFB import build_net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    net = build_net('test', 300, 3, 3, 'FPN')  # initialize detector

    input = torch.randn(1, 3, 300, 300)
    flops, params = profile(net, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print("par.: ", params)
    print("Mult.: ", flops)

    params_num = count_parameters(net)
    print("par.:  {}K".format(params_num/1000.0))

    net = net.cuda()
    net.eval()

    data_tensor = torch.rand(1, 3, 300, 300).cuda()
    start = time.perf_counter()
    for _ in tqdm(range(100)):
        _ = net(data_tensor)
    end = time.perf_counter()
    print("average forward time= {:0.5f}s".format((end - start)/100))

if __name__ == "__main__":
    main()