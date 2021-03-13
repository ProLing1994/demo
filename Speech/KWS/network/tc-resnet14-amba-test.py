import importlib
import torch
import sys
sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')

if __name__ == '__main__':

    net_module = importlib.import_module('network.' + 'tc-resnet14-amba')
    net = net_module.__getattribute__('SpeechResModel')(3, 201, 40)
    
    checkpoint = torch.load("/mnt/huanyuan/model/audio_model/kws_xiaorui_tc_resnet14/net_parameter.pth")
    net.load_state_dict(checkpoint)

    net.eval()
    input = torch.ones([1, 1, 201, 40])
    output = net(input)
    print(output)