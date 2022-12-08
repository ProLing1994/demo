import importlib
import numpy as np
import torch
import torch.nn as nn
import sys

from thop import profile   
from thop import clever_format
sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')


def test_vdnet_receptive_field_size(model, inputs, outputs, patch_dimension):
    dim_x = patch_dimension
    dim_y = patch_dimension

    output_center = outputs.data[0, :, outputs.shape[2] // 2, outputs.shape[3] // 2]

    no_changes_yet = True
    i = 0
    while i <= patch_dimension // 2 and no_changes_yet:

        inputs[0, 0, i, i:dim_y - i] = inputs[0, 0, i, i:dim_y - i] + np.random.normal()

        inputs[0, 0, dim_x - 1 - i, i:dim_y - i] = inputs[0, 0, dim_x - 1 - i, i:dim_y - i] + np.random.normal()

        inputs[0, 0, i:dim_x - i, i] = inputs[0, 0, i:dim_x - i, i] + np.random.normal()

        inputs[0, 0, i:dim_x - i, dim_y - 1 - i] = inputs[0, 0, i:dim_x - i, dim_y - 1 - i] + np.random.normal()

        outputs = model(inputs)
        new_output_center = outputs.data[0, :, outputs.shape[2] // 2, outputs.shape[3] // 2]

        i += 1

        if torch.max(torch.abs(new_output_center - output_center)):
            print("The model receptive field size is empirically greater than or equal to {0}".format(patch_dimension - 2 * i))
            no_changes_yet = False


def main():
    batch_size = 1
    in_channels = 1
    patch_dimension = 501
    image_height = patch_dimension
    image_weidth = patch_dimension
    num_classes = 3

    # net_name = "res15"
    net_name = "tc-resnet14-dropout-receptive-field-test"

    # load network structure
    net_module = importlib.import_module('network.' + net_name)
    net = net_module.SpeechResModel(num_classes=num_classes, 
                                    image_height=image_height, 
                                    image_weidth=image_weidth)
    net_module.parameters_init(net)
    model = nn.parallel.DataParallel(net, device_ids=[0])
    model = model.cuda()
    model.eval()

    inputs = torch.randn(batch_size, in_channels, image_height, image_weidth)
    inputs = torch.autograd.Variable(inputs)
    inputs = inputs.cuda()
    outputs = model(inputs)

    # test_receptive_field_size
    test_vdnet_receptive_field_size(model, inputs, outputs, patch_dimension)


if __name__ == "__main__":
    main()