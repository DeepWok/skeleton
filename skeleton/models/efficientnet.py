import functools

import torch
from torch import nn
from torch.nn import functional as F

from ..utils import device
from .common import Sequential, Conv2dSame, BatchNorm2d, NoConnection, Identity
from .base import ModelBase


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    # uniform [0,1)
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MBNoConnection(NoConnection):
    def __init__(
            self, inputs=None, outputs=None, stride=1, affine=True,
            original_expand=1,
            kernel=None, padding=None, expansion=1, groups=1,
            se_rate=0.25, training=True, drop_connect_rate=0.0):
        super().__init__(inputs, outputs, stride)


class IdentityOrReduce(nn.Module):
    def __init__(
            self, inputs=None, outputs=None, kernel=1, stride=None, search=True, **kwargs):
        super().__init__()
        if stride == 1 and inputs==outputs:
            self.layer = Identity()
        else:
            self.layer = Conv2dSame(inputs, outputs, 1, stride)

    def forward(self, x):
        return self.layer(x)



class MBConvBlock(nn.Module):
    def __init__(
            self, inputs=None, outputs=None, stride=1, affine=True,
            original_expand=1, kernel=None, padding=None,
            expansion=1, groups=1,
            se_rate=0.25, training=True, use_shortcut=False):
        super().__init__()
        self.se = se_rate is not None
        self.training = training

        pre_expand_bn = BatchNorm2d(inputs)
        # expansion phase
        expanded = inputs * expansion * original_expand
        expand_layers = []
        # if expanded != inputs:
        expand_conv = Conv2dSame(
            inputs, expanded, 1, 1, groups=groups, bias=False)
        expand_bn = BatchNorm2d(expanded)
        expand_layers = [
            pre_expand_bn, nn.ReLU(), expand_conv, expand_bn, nn.ReLU()]

        if padding is None:
            padding = {1: 0, 3: 1, 5: 2, 7: 3}[kernel]

        # depthwise layers
        depthwise_layers = [
            Conv2dSame(
                expanded, expanded, kernel, stride, padding,
                groups=expanded, bias=False),
            BatchNorm2d(expanded),
            nn.ReLU()
        ]

        # squeeze and excitation
        se_layers = []
        if self.se:
            squeezed = max(1, int(inputs*se_rate))
            se_layers = [
                Conv2dSame(outputs, squeezed, 1, 1),
                nn.ReLU(),
                # Swish(),
                Conv2dSame(squeezed, outputs, 1, 1),
            ]

        # pointwise layer
        pointwise_layer = [
            Conv2dSame(expanded, outputs, 1, 1, groups=groups, bias=False),
            # BatchNorm2d(outputs, affine=affine)
        ]

        self.expand_layers = nn.Sequential(*expand_layers)
        self.dw_layers = nn.Sequential(*depthwise_layers)
        self.se_layers = nn.Sequential(*se_layers)
        self.pw_layers = nn.Sequential(*pointwise_layer)

        self.stride = stride
        if stride == 1 and inputs == outputs and use_shortcut:
            self.shortcut = True
        else:
            self.shortcut = False

    def forward(self, x):
        inputs = x
        x = self.expand_layers(x)
        x = self.dw_layers(x)
        x = self.pw_layers(x)
        if self.se:
            squeezed = F.adaptive_avg_pool2d(x, 1)
            squeezed = self.se_layers(x)
            x = torch.sigmoid(squeezed) * x
        if self.shortcut:
            x = inputs + x
        return x


TYPES = {
    'identity': IdentityOrReduce,
    'k1_e1': functools.partial(MBConvBlock, kernel=1, groups=1, expansion=1),
    'k1_e1_g2': functools.partial(MBConvBlock, kernel=1, groups=2, expansion=1),
    'k1_e2': functools.partial(MBConvBlock, kernel=1, groups=1, expansion=2),
    # 'k1_e4': functools.partial(MBConvBlock, kernel=1, groups=1, expansion=4),
    'k3_e1': functools.partial(MBConvBlock, kernel=3, groups=1, expansion=1),
    'k3_e1_g2': functools.partial(MBConvBlock, kernel=3, groups=2, expansion=1),
    'k3_e2': functools.partial(MBConvBlock, kernel=3, groups=1, expansion=2),
    # 'k3_e4': functools.partial(MBConvBlock, kernel=3, groups=1, expansion=4),
    'k5_e1': functools.partial(MBConvBlock, kernel=5, groups=1, expansion=1),
    'k5_e1_g2': functools.partial(MBConvBlock, kernel=5, groups=2, expansion=1),
    'k5_e2': functools.partial(MBConvBlock, kernel=5, groups=1, expansion=2),
    # 'k5_e4': functools.partial(MBConvBlock, kernel=5, groups=1, expansion=4),
}


class EfficientMixture(nn.Module):
    def __init__(
            self, mixture_type,
            mask=None, inputs=None, outputs=None, stride=None,
            original_expand=None, drop_connect_rate=None,
            se_rate=None, first=False):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        additional_op_args = {
            'original_expand': original_expand,
            'se_rate': se_rate,
        }
        self.mixture = mixture_type(
            TYPES, mask, inputs, outputs, stride,
            first=first, additional_op_args=additional_op_args)

        if stride == 1 and inputs == outputs:
            self.shortcut = True
        else:
            self.shortcut = False
            self.reduce = Conv2dSame(inputs, outputs, 1, stride=stride, groups=1)

    def forward(self, x, i=None):
        m, i = self.mixture(x, i)
        if self.shortcut:
            m = m + x
        else:
            m = m + self.reduce(x)
        return m, i


class EfficientNet(ModelBase):
    # EfficientNet B0 style
    # https: // arxiv.org/abs/1905.11946
    # https: // github.com/lukemelas/EfficientNet-PyTorch

    name = 'efficientnet'
    dropout_rate = 0.2
    drop_connect_rate = 0.2
    se_rate = 0.25
    first_stride = 2
    filters = [16, 24, 40, 80, 112, 192, 320]
    strides = [1, 2, 2, 2, 1, 2, 1]
    repeats = [1, 2, 2, 3, 3, 4, 1]
    original_expand = [1, 6, 6, 6, 6, 6, 6]

    def __init__(self, num_classes, mixture_type, masks=None):
        super().__init__(num_classes, mixture_type, masks)
        # Stem
        inputs = 32
        self.conv0 = Sequential(
            Conv2dSame(
                3, inputs,
                kernel_size=3,
                stride=self.first_stride, bias=False),
            BatchNorm2d(32),
            nn.ReLU(),
        )
        first = True
        layers = []
        stages = zip(
            self.filters, self.strides, self.repeats, self.original_expand)

        for outputs, stride, repeat, original_expand in stages:
            for _ in range(repeat):
                mask = self.pop_mask(len(TYPES))
                mixture = EfficientMixture(
                    mixture_type, mask, inputs, outputs, stride,
                    original_expand, self.drop_connect_rate,
                    self.se_rate, first)
                layers.append(mixture)
                first = False
                inputs = outputs
                stride = 1

        self.layers = Sequential(*layers)
        # pooling and fc
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(320, num_classes)

    def forward(self, x):
        conved = self.conv0(x)
        stacked = self.layers(conved)
        if isinstance(stacked, tuple):
            stacked, _ = stacked
        pooled = self.pool(stacked)
        return self.fc(pooled.squeeze())


class EfficientNetSmall(EfficientNet):
    # EfficientNet B0, cifar10 style
    name = 'efficientnetsmall'
    dropout_rate = 0.5
    drop_connect_rate = 0.0

    first_stride = 1
    filters = [16, 24, 40, 80, 112, 192, 320]
    repeats = [1, 2, 2, 2, 2, 2, 1]
    strides = [1, 2, 1, 2, 1, 2, 1]
    original_expand = [1, 3, 3, 3, 3, 3, 3]


class EfficientNetB0(EfficientNetSmall):
    name = 'efficientnetb0'
    mbblock = functools.partial(
            MBConvBlock, kernel=3, groups=1, expansion=1, use_shortcut=True)

    def __init__(self, num_classes, mixture_type, masks=None):
        super().__init__(num_classes, mixture_type, masks)
        # Stem
        inputs = 32
        self.conv0 = Sequential(
            Conv2dSame(
                3, inputs,
                kernel_size=3,
                stride=self.first_stride, bias=False),
            BatchNorm2d(32),
            nn.ReLU(),
        )
        first = True
        layers = []
        stages = zip(
            self.filters, self.strides, self.repeats, self.original_expand)

        for outputs, stride, repeat, expand in stages:
            for _ in range(repeat):
                mask = self.pop_mask(len(TYPES))
                layer = self.mbblock(
                    inputs=inputs, outputs=outputs, stride=stride, original_expand=expand)
                layers.append(layer)
                first = False
                inputs = outputs

        self.layers = Sequential(*layers)
        # pooling and fc
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(320, num_classes)
