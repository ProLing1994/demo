import importlib
import sys
import torch
import tensorwatch as tw
import torchvision.models

from torchviz import make_dot

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')


def main():
    # TO Do
    net_name = "res15"
    # net_name = "res15-narrow"

    # load network structure
    net_module = importlib.import_module('network.' + net_name)
    net = net_module.SpeechResModel(num_classes=3, 
                                    image_height=201, 
                                    image_weidth=40)
    model_tree = tw.draw_model(net, [1, 1, 201, 40], orientation='TB')
    model_tree.save("/home/huanyuan/code/demo/Speech/KWS/visualization/model/model.jpg")

    print(tw.model_stats(net, [1, 1, 201, 40])) 
if __name__ == "__main__":
    main()