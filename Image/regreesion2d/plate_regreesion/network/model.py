import torch.nn as nn
import torch


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu_layer = torch.nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu_layer(x - 3) / 6


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.hard_sigmoid_layer = HardSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.hard_sigmoid_layer(x)


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid_layer(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedDepthPointBlock(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedDepthPointBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class InvertedResidualV3(nn.Module):
    def __init__(self, inp, oup, stride, expand_size, use_se=False, kernel_size=3, activation_function='relu6'):
        super(InvertedResidualV3, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert activation_function in ['relu', 'relu6', 'hard_swish', 'swish']
        hidden_dim = expand_size
        self.use_res_connect = self.stride == 1 and inp == oup
        if activation_function == 'relu6':
            self.activation_function = nn.ReLU6(inplace=True)
        elif activation_function == 'hard_swish':
            self.activation_function = HardSwish(inplace=True)
        elif activation_function == 'swish':
            self.activation_function = Swish(inplace=True)
        else:
            self.activation_function = nn.ReLU(inplace=True)
        self.se_layer = SELayer(hidden_dim, reduction=8) if use_se else None
        self.conv = [
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            self.activation_function,
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=hidden_dim,
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            self.activation_function, ]
        self.conv = nn.Sequential(*self.conv)
        self.conv2 = [nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                      nn.BatchNorm2d(oup)]
        self.conv2 = nn.Sequential(*self.conv2)

    def forward(self, x):
        tmp = self.conv(x)
        if self.se_layer:
            tmp = self.se_layer(tmp)
        tmp = self.conv2(tmp)
        if self.use_res_connect:
            return x + tmp
        else:
            return tmp


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, width_mult=1., input_channel=32, last_channel=1280,
                 interverted_residual_setting=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        if interverted_residual_setting is None:
            interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class MobileNetV1(nn.Module):
    def __init__(self, n_class=1000, width_mult=1., input_channel=32, last_channel=1024,
                 interverted_residual_setting=None):
        super(MobileNetV1, self).__init__()
        block = InvertedDepthPointBlock
        if interverted_residual_setting is None:
            interverted_residual_setting = [
                # c, n, s
                [64, 1, 1],
                [128, 2, 2],
                [256, 2, 2],
                [512, 6, 2],
                [1024, 2, 2],
            ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s))
                else:
                    self.features.append(block(input_channel, output_channel, 1))
                input_channel = output_channel
        # make it nn.Sequential
        if self.last_channel != input_channel:
            self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, width_mult=1., input_channel=16, last_channel=1280,
                 interverted_residual_setting=None):
        super(MobileNetV3, self).__init__()
        block = InvertedResidualV3
        if interverted_residual_setting is None:
            interverted_residual_setting = [
                # expand size, c, s, a, kernel size, use_se
                [16, 16, 2, 'relu6', 3, True],
                [72, 24, 2, 'relu6', 3, False],
                [88, 24, 1, 'relu6', 3, False],
                [96, 40, 2, 'hard_swish', 5, True],
                [240, 40, 1, 'hard_swish', 5, True],
                [240, 40, 1, 'hard_swish', 5, True],
                [120, 48, 1, 'hard_swish', 5, True],
                [144, 48, 1, 'hard_swish', 5, True],
                [288, 96, 2, 'hard_swish', 5, True],
                [576, 96, 1, 'hard_swish', 5, True],
                [576, 96, 1, 'hard_swish', 5, True],
            ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # building inverted residual blocks
        for e, c, s, a, k, use_se in interverted_residual_setting:
            output_channel = int(c * width_mult)
            self.features.append(
                block(input_channel, output_channel, s, e, use_se=use_se, kernel_size=k, activation_function=a))
            input_channel = output_channel
        # building last several layers
        # self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
            Swish(inplace=True),
            nn.Conv2d(self.last_channel, n_class, 1, 1, 0, bias=False),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.classifier(x)
        b, c, _, _ = x.size()
        x = x.view(b, c)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
