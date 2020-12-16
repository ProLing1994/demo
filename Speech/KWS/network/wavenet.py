  
import torch
from torch import nn

def parameters_init(net):
    pass

class WaveNetConv(nn.Module):
    #            |----------------------------------------|     *residual*
    #            |                                        |
    #            |    |-- conv -- tanh --|                |
    # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
    #                 |-- conv -- sigm --|     |
    #                                         1x1
    #                                          |
    # ---------------------------------------> + ------------->	*skip*

    def __init__(self, num_features_in, num_features_out, skip_channels, kernel_size, dilation):
        self.kernel_size = kernel_size
        super(WaveNetConv, self).__init__()
        self.conv_tanh = nn.Sequential(*[nn.Conv1d(num_features_in, num_features_out, kernel_size, dilation=dilation),
                                         nn.Tanh()])
        self.conv_sig = nn.Sequential(*[nn.Conv1d(num_features_in, num_features_out, kernel_size, dilation=dilation),
                                        nn.Sigmoid()])
        self.conv_residual = nn.Conv1d(num_features_out, num_features_out, 1, dilation=dilation)
        self.conv_skip = nn.Conv1d(num_features_out, skip_channels, 1, dilation=dilation)
        self.norm = nn.BatchNorm1d(num_features_in)

    def forward(self, x, skip):
        '''
        :param x: [batch,  features, timesteps,]
        '''
        x = self.norm(x)
        x_ = self.conv_tanh(x) * self.conv_sig(x)
        x_skip = self.conv_skip(x_)
        x_ = self.conv_residual(x_)
        if x_.shape[-1] != x.shape[-1]:
            padding = int((x.shape[-1] - x_.shape[-1]) // 2)
            x_ = x[:, :, padding:-padding] + x_
            skip = skip[:, :, padding:-padding] + x_skip
        else:
            x_ = x + x_
            skip = skip + x_skip
        return x_, skip


class WaveNetBlock(nn.Module):
    def __init__(self, num_features_in, num_features_out, skip_channels, kernel_size=3, dilations=[1, 2, 4, 8]):
        self.kernel_size = kernel_size
        super(WaveNetBlock, self).__init__()
        self.convs = nn.ModuleList([WaveNetConv(num_features_in, num_features_out, skip_channels, kernel_size, dilation)
                                    for dilation in dilations])

    def forward(self, x, skip):
        '''
        :param x: [batch, timesteps, features]
        '''
        for idx, conv in enumerate(self.convs):
            x, skip = conv(x, skip)
        return x, skip


class SpeechResModel(nn.Module):
    def __init__(self, num_classes, image_height, image_weidth, num_blocks=6, residual_channels=16, skip_channels=32, kernel_size=3,
                 dilations=[1, 2, 4, 8]):
        super().__init__()
        self.skip_channels = skip_channels
        self.start_conv = nn.Conv1d(image_weidth, residual_channels, 1)
        self.blocks = nn.ModuleList([
            WaveNetBlock(residual_channels, residual_channels, skip_channels, kernel_size, dilations)
            for block_idx in range(num_blocks)])
        self.classifer = nn.Sequential(
            *[nn.ReLU(), nn.Conv1d(skip_channels, skip_channels, 1), nn.ReLU(), nn.Conv1d(skip_channels, num_classes, 1)])

    def forward(self, x: torch.tensor):
        '''
        :param x: [batch, channels, timesteps, features]
        :return:
        '''
        x = x.view(x.size(0), x.size(2), x.size(3))         # shape: (batch, 1, 201, 40) ->  # shape: (batch, 201, 40)
        x = x.transpose(1, 2).float()                       # shape: (batch, 201, 40) ->  # shape: (batch, 40, 201)
        x = self.start_conv(x)                              # shape: (batch, 40, 201) ->  # shape: (batch, 16, 201)
        skip = torch.zeros((x.shape[0], int(self.skip_channels), x.shape[-1])).to(x.device)
        for idx, block in enumerate(self.blocks):
            x, skip = block(x, skip)
        out = self.classifer(skip)
        out = torch.mean(out, 2)                            # shape: (batch, 3, 21) ->  # shape: (batch, 3)
        return out
