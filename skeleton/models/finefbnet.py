import functools

import torch
from torch import nn

from .common import Sequential, AdaptiveIdentity, ChannelShuffle
from .base import ModelBase


class Conv(nn.Module):
    def __init__(
            self, inputs, outputs, kernel,
            stride=1, padding=None, position='head', expansion=1, groups=1,
            normalizer=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.expanded = inputs * expansion
        if padding is None:
            padding = {1: 0, 3: 1, 5: 2, 7: 3}[kernel]
        if position == 'head':
            self.layers = nn.Sequential(
                nn.Conv2d(
                    inputs, self.expanded, kernel, stride,
                    padding=padding, bias=False),
                nn.BatchNorm2d(self.expanded),
                act())
        elif position == 'mid':
            self.layers = nn.Sequential(
                nn.Conv2d(
                    self.expanded, self.expanded, 1, 1,
                    bias=False),
                nn.BatchNorm2d(self.expanded),
                act())
        elif position == 'tail':
            self.layers = nn.Sequential(
                nn.Conv2d(
                    self.expanded, outputs, 1, 1, bias=False),
                nn.BatchNorm2d(outputs),
                act())
        else:
            raise ValueError(f'Unrecognized position name {position!r}.')


    def forward(self, x):
        return self.layers(x)


def get_types():
    args = {
        'none': (None, None, None),
        'k1_e1': (1, 1, 1),
        'k1_e1_g2': (1, 1, 2),
        # 'k1_e1_g4': (1, 1, 4),
        'k3_e1': (3, 1, 1),
        'k3_e1_g2': (3, 1, 2),
        # 'k3_e1_g4': (3, 1, 4),
        'k5_e1': (5, 1, 1),
        'k5_e1_g2': (5, 1, 2),
        # 'k5_e1_g4': (5, 1, 4),
    }
    head, mid, tail = {}, {}, {}
    for key, (kernel, expansion, groups) in args.items():
        if key == 'none':
            head[key] = mid[key] = tail[key] = AdaptiveIdentity
            continue
        for name, position in (('head', head), ('mid', mid), ('tail', tail)):
            position[key] = functools.partial(
                Conv, position=name,
                kernel=kernel, groups=groups, expansion=expansion)
    return {'head': head, 'mid': mid, 'tail': tail}


class Block(nn.Module):
    def __init__(self, inputs, outputs, stride, types, mixture_type):
        super().__init__()
        instantiated = {}
        for n, p in types.items():
            s = stride if n == 'head' else 1
            o = outputs if n == 'tail' else inputs
            instantiated[n] = {k: t(inputs, o, stride=s) for k, t in p.items()}
        self.layers = Sequential(
            mixture_type(inputs, inputs, instantiated['head'], first=True),
            mixture_type(inputs, inputs, instantiated['mid'], first=False),
            mixture_type(inputs, outputs, instantiated['tail'], first=False)
        )
        self.num_ops = len(instantiated['tail'].keys())
        self.shortcut = AdaptiveIdentity(
            inputs, outputs, stride, norm_act=True)

    def forward(self, x):
        return self.layers(x)[0] + self.shortcut(x)


class FineFBNet(ModelBase):
    name = 'FineFBNet'
    conv0_stride = 2
    stages = [
        (16, 1, 1), (24, 2, 4), (32, 2, 4), (64, 2, 4), (112, 1, 4),
        (184, 2, 4), (352, 2, 1), (352, 2, 1)]

    def __init__(self, num_classes, mixture_type):
        super().__init__(num_classes)
        inputs = 16
        self.conv0 = Sequential(
            torch.nn.Conv2d(
                3, inputs, kernel_size=3, stride=self.conv0_stride,
                padding=1, bias=False),
            torch.nn.BatchNorm2d(inputs))
        types = get_types()
        layers = []
        for outputs, stride, repeat in self.stages:
            for _ in range(repeat):
                layers.append(
                    Block(inputs, outputs, stride, types, mixture_type))
                inputs = outputs
                stride = 1
        self.layers = Sequential(*layers)
        # pooling and fc
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(self.stages[-1][0], num_classes)

    def forward(self, x):
        conved = self.conv0(x)
        stacked = self.layers(conved)
        pooled = self.pool(stacked)
        return self.fc(pooled.squeeze())


class FineFBNetSmall(FineFBNet):
    name = 'FineFBNet'
    conv0_stride = 1
    stages = [
        (16, 1, 1), (24, 1, 1), (32, 2, 1), (64, 2, 1), (112, 1, 1),
        (184, 2, 1), (352, 1, 1)]
