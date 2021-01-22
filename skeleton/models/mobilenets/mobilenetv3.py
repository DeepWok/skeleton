import torch
import math
from torch import nn
from flareon.models.base import ModelBase
from .util import (
    _make_divisible, h_swish, h_sigmoid, conv_1x1_bn, conv_3x3_bn,
    InvertedResidual)


class MobileNetV3(ModelBase):
    default_cfgs = [
        # k, t, c, SE, HS, s
        [3, 16,  16,  False, 'RE', 1],
        [3, 64,  24,  False, 'RE', 1], # adapt cifar10
        [3, 72,  24,  False, 'RE', 1],
        [5, 72,  40,  True,  'RE', 2],
        [5, 120, 40,  True,  'RE', 1],
        [5, 120, 40,  True,  'RE', 1],
        [3, 240, 80,  False, 'HS', 2],
        [3, 200, 80,  False, 'HS', 1],
        [3, 184, 80,  False, 'HS', 1],
        [3, 184, 80,  False, 'HS', 1],
        [3, 480, 112, True,  'HS', 1],
        [3, 672, 112, True,  'HS', 1],
        [5, 672, 160, True,  'HS', 2],
        [5, 960, 160, True,  'HS', 1],
        [5, 960, 160, True,  'HS', 1],
    ]
    mode = 'large'
    width_mult = 1.
    def __init__(self, num_classes=1000, *args, **kwargs):
        super().__init__(num_classes, *args, **kwargs)
        # setting of inverted residual blocks
        mode = self.mode
        assert mode in ['large', 'small']
        self.cfgs = kwargs.pop("cfgs", self.default_cfgs)
        print(self.cfgs)

        width_mult = self.width_mult
        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 1)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, exp, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(exp * width_mult, 8)
            layers.append(
                block(
                    input_channel,
                    exp_size,
                    output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.8),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNetV3Small(MobileNetV3):
    default_cfgs = [
        [3, 16,  16,  True,  'RE', 1], # adapt cifar10
        [3, 72,  24,  False, 'RE', 2],
        [3, 88,  24,  False, 'RE', 1],
        [5, 96,  40,  True,  'HS', 2],
        [5, 240, 40,  True,  'HS', 1],
        [5, 240, 40,  True,  'HS', 1],
        [5, 120, 48,  True,  'HS', 1],
        [5, 144, 48,  True,  'HS', 1],
        [5, 288, 96,  True,  'HS', 2],
        [5, 576, 96,  True,  'HS', 1],
        [5, 576, 96,  True,  'HS', 1],
    ]
    mode = 'small'
