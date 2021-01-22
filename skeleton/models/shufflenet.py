import functools

import torch
from torch import nn

from .base import ModelBase
from .common import Sequential, AdaptiveIdentity


class ChoiceBlock(nn.Module):
    def __init__(
            self, inputs, outputs, stride=None, kernel=None, paddding=None,
            expansion=1, stacked=False):
        super().__init__()
        if outputs is not None:
            expanded = outputs * expansion

        if stride == 1 and inputs == outputs:
            shortcut = nn.Sequential()
        else:
            shortcut = nn.Sequential(
                nn.Conv2d(
                    inputs, inputs,
                    kernel_size=kernel, stride=stride, groups=inputs),
                nn.GroupNorm(num_channels=expanded, num_groups=16),
                nn.ReLU(),
                nn.Conv2d(expanded, outputs, kernel_size=1))
                # nn.GroupNorm(num_channels=channels, num_groups=16),
                # nn.ReLU())

        if stacked:
            layers = nn.Sequential(
                # DW
                nn.Conv2d(
                    inputs, inputs,
                    kernel_size=kernel, stride=stride, groups=inputs),
                nn.GroupNorm(num_channels=expanded, num_groups=16),
                nn.ReLU(),
                # 1*1
                nn.Conv2d(expanded, expanded, kernel_size=1),
                nn.GroupNorm(num_channels=expanded, num_groups=16),
                nn.ReLU(),
                # DW
                nn.Conv2d(
                    expanded, expanded,
                    kernel_size=kernel, stride=1, groups=expanded),
                nn.GroupNorm(num_channels=expanded, num_groups=16),
                nn.ReLU(),
                # 1*1
                nn.Conv2d(expanded, expanded, kernel_size=1),
                nn.GroupNorm(num_channels=expanded, num_groups=16),
                nn.ReLU(),
                # DW
                nn.Conv2d(
                    expanded, expanded,
                    kernel_size=kernel, stride=1, groups=expanded),
                nn.GroupNorm(num_channels=expanded, num_groups=16),
                nn.ReLU(),
                # 1*1
                nn.Conv2d(expanded, outputs, kernel_size=1))
                # nn.BatchNorm2d(channels),
                # nn.ReLU())
        else:
            layers = nn.Sequential(
                # 1*1
                nn.Conv2d(inputs, expanded, kernel_size=1),
                nn.GroupNorm(num_channels=expanded, num_groups=16),
                nn.ReLU(),
                # DW
                nn.Conv2d(
                    expanded, expanded,
                    kernel_size=kernel, stride=stride, groups=expanded),
                nn.GroupNorm(num_channels=expanded, num_groups=16),
                nn.ReLU(),
                # 1*1
                nn.Conv2d(expanded, outputs, kernel_size=1))
        self.layers = layers
        self.shortcut = shortcut
        # self.norm_act = nn.Sequential(
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU())

    def forward(self, x):
        tmp = self.layers(x) + self.shortcut(x)
        print(tmp.shape)
        return tmp


TYPES = {
    'none': AdaptiveIdentity,
    'k3_s1': functools.partial(ChoiceBlock, kernel=3, stride=1),
    'k3_s2': functools.partial(ChoiceBlock, kernel=3, stride=2),
    'k5_s1': functools.partial(ChoiceBlock, kernel=5, stride=1),
    'k5_s2': functools.partial(ChoiceBlock, kernel=5, stride=2),
    'k7_s1': functools.partial(ChoiceBlock, kernel=7, stride=1),
    'k7_s2': functools.partial(ChoiceBlock, kernel=7, stride=2),
    'k3_s1_stacked': functools.partial(ChoiceBlock, kernel=3, stacked=True),
    'k3_s2_stacked': functools.partial(
        ChoiceBlock, kernel=3, stride=2, stacked=True),
}


class ShuffleNet(ModelBase):
    def __init__(self, num_classes, mixture_type):
        super().__init__(num_classes)
        self.mixture_type = mixture_type
        self.layers = []
        self.conv0 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=0)
        # stage 0
        self._multi_block_instantiate(16, 64, 4, 's0', 2)
        # stage 1
        self._multi_block_instantiate(64, 160, 4, 's1', 2)
        # stage 2
        self._multi_block_instantiate(160, 320, 8, 's3', 2)
        # stage 3
        self._multi_block_instantiate(320, 640, 4, 's3', 2)
        self.layers = Sequential(*self.layers)
        self.conv4 = torch.nn.Conv2d(640, 1024, kernel_size=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        # self.layers.append(self.pool)
        self.fc = torch.nn.Linear(1024, num_classes)

    def _multi_block_instantiate(
            self, inputs, outputs, stride, name_prefix, nblocks):
        for _ in range(nblocks):
            ops = {
                k: t(inputs=inputs, outputs=outputs) for k, t in TYPES.items()}
            self.layers.append(self.mixture_type(inputs, outputs, ops))
            inputs = outputs
            stride = 1

    def forward(self, x):
        conved = self.conv0(x)
        stacked, _ = self.layers(conved)
        x = self.conv4(stacked)
        pooled = self.pool(x)
        return self.fc(pooled.squeeze())
