import functools

import torch
from torch import nn
import numpy as np

from ..utils import torch_cuda


BatchNorm2d = functools.partial(nn.BatchNorm2d)


class Sequential(nn.ModuleList):
    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, *args):
        for m in self:
            args = m(*args)
            if not isinstance(args, tuple):
                args = (args, )
        if len(args) == 1:
            return args[0]
        return args

    def estimate(self, inputs, outputs):
        return {'#macs': 0}


class Linear(nn.Linear):
    def estimate(self, inputs, outputs):
        cin = inputs[0].size(1)
        cout = outputs.size(1)
        return {'#macs': int(cin * cout)}


class Conv2d(nn.Conv2d):
    def estimate(self, inputs, outputs):
        kh, kw = self.kernel_size
        cin = inputs[0].size(1)
        _, cout, hout, wout = outputs.shape
        return {'#macs': int(np.product([kh, kw, cin, cout, hout, wout]))}


class Conv2dSame(Conv2d):
    def __init__(
            self, in_channels, out_channels, kernel_size,
            stride=1, padding=None, dilation=1, groups=1,
            bias=True):
        if padding is None:
            try:
                padding = {1: 0, 3: 1, 5: 2, 7: 3}[kernel_size]
            except KeyError:
                raise ValueError(
                    f'Unsupported padding for kernel size {kernel_size}.')
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)


class AdaptiveIdentity(nn.Module):
    def __init__(
            self, inputs=None, outputs=None, stride=None,
            norm=False, affine=True, relu=False):
        super().__init__()
        layer = None
        if inputs != outputs:
            layers = [Conv2dSame(inputs, outputs, 1, stride, bias=not norm)]
            if norm:
                layers.append(BatchNorm2d(outputs, affine=affine))
            if relu:
                layers.append(nn.ReLU(inplace=True))
            layer = nn.Sequential(*layers)
        elif stride != 1:
            layer = nn.AvgPool2d(1, stride=stride)
        self.layer = layer

    def forward(self, x):
        if not self.layer:
            return x
        return self.layer(x)


class NoConnection(nn.Module):
    def __init__(self, inputs=None, outputs=None, stride=None):
        super().__init__()
        self.stride = stride
        self.outputs = outputs

    def forward(self, x):
        batch, _, height, width = x.size()
        if self.stride == 2:
            if height % 2 or width % 2:
                raise ValueError(
                    f'Resolution ({height}, {width}) not divisible by 2')
            height = int(height / 2)
            width = int(width / 2)
        shape = (batch, self.outputs, height, width)
        return torch_cuda.FloatTensor(*shape).fill_(0)


class SqueezeExcite(nn.Module):
    def __init__(self, channels, factor=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, int(channels / factor), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channels / factor), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, ic, _, _ = x.size()
        y = self.avg_pool(x).view(b, ic)
        y = self.fc(y).view(b, ic, 1, 1)
        # multiply by scaling
        return x * y.expand_as(x)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        if self.groups == 1:
            return x
        batch, channels, height, width = x.data.size()
        if channels % self.groups:
            raise ValueError(
                f'The number of channels {channels} must be divisible '
                f'by the number of groups {self.groups}.')
        x = x.view(batch, self.groups, channels // self.groups, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch, channels, height, width)
        return x


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class PoolBase(nn.Module):
    def __init__(
            self, kernel=3, stride=1, padding=1,
            inputs=None, search=True, **kwargs):
        super().__init__()
        self.layer = self._pool(kernel, stride, padding)
        if search:
            self.layer = torch.nn.Sequential(
                self.layer, torch.nn.BatchNorm2d(inputs, affine=False))

    def _pool(self, kernel, stride, padding):
        raise NotImplementedError

    def forward(self, x):
        return self.layer(x)


class AvgPool(PoolBase):
    def _pool(self, kernel, stride, padding):
        return torch.nn.AvgPool2d(
            kernel, stride=stride, padding=padding, count_include_pad=False)


class MaxPool(PoolBase):
    def _pool(self, kernel, stride, padding):
        return torch.nn.MaxPool2d(kernel, stride=stride, padding=padding)


class Identity(nn.Module):
    def forward(self, x):
        return x
