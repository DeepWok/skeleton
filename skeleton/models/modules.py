import functools
import math
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

from collections import OrderedDict

from flareon.utils import device
from .common import (
    Sequential, BatchNorm2d, AdaptiveIdentity,
    ChannelShuffle, NoConnection)
from .mobilenets.util import _make_divisible, sub_filter_start_end, h_sigmoid, h_swish


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class DynamicBatchNorm2d(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, feature_dims):
        super(DynamicBatchNorm2d, self).__init__()

        self.max_feature_dim = max(feature_dims)
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x, bn.running_mean[:feature_dim], bn.running_var[:feature_dim], bn.weight[:feature_dim],
                bn.bias[:feature_dim], bn.training or not bn.track_running_stats,
                exponential_average_factor, bn.eps,
            )

    def forward(self, x):
        self.image_shape = tuple(x.shape)
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y

    def get_num_parameters_macs(self):
        macs = np.prod(np.array(self.image_shape))
        return 0, macs


class Conv2dSame(torch.nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_sizes,
            stride=1, dilation=1, groups=1,
            bias=False):
        super(Conv2dSame, self).__init__()
        # super(Conv2dSame, self).__init__(
        #     max(in_channels), max(out_channels), max(kernel_sizes),
        #     stride=1, padding=0, dilation=1, groups=1,
        #     bias=bias, padding_mode='zeros')
        max_in = max(in_channels)
        max_out = max(out_channels)
        max_k = (max(kernel_sizes), max(kernel_sizes))
        self.weight = Parameter(torch.Tensor(
            max_out, max_in // groups, *max_k))
        if bias:
            self.bias = Parameter(torch.Tensor(max_out))
        else:
            self.register_parameter('bias', None)

        self.padding_config = {1: 0, 3: 1, 5: 2, 7: 3}
        self.max_config = {
            'ic': max(in_channels),
            'oc': max(out_channels), 'k': max(kernel_sizes)}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _slice_weight(self, w, ic, oc, k):
        assert k <= self.max_config['k']
        assert ic <= self.max_config['ic']
        assert oc <= self.max_config['oc']
        # take the middle parts
        k_s, k_e = sub_filter_start_end(self.max_config['k'], k)
        w = w[:oc, :ic, k_s:k_e, k_s:k_e]
        return w

    def forward(self, image, ic_index, oc_index, k_index):
        self.image_shape = tuple(image.shape)
        ic = self.in_channels[ic_index]
        oc = self.out_channels[oc_index]
        k = self.kernel_sizes[k_index]
        padding = self.padding_config[k]
        weight = self._slice_weight(
            self.weight, ic, oc, k)
        return F.conv2d(
            image, weight, None, self.stride, padding, self.dilation, 1)

    def get_num_parameters_macs(self, ic_index, oc_index, k_index):
        _, c, h, w = self.image_shape
        ic = self.in_channels[ic_index]
        oc = self.out_channels[oc_index]
        k = self.kernel_sizes[k_index]
        params = k * k * ic * oc
        macs = k * k * ic * oc * (h / self.stride) * (w / self.stride)
        macs = int(macs)
        return params, macs


class DepthwiseConv2dSame(Conv2dSame):
    def __init__(
            self, in_channels, kernel_sizes,
            stride=1, dilation=1,
            bias=False):
        super(DepthwiseConv2dSame, self).__init__(
            in_channels, in_channels, kernel_sizes,
            stride=stride, dilation=dilation)
        self.groups = max(in_channels)

    def forward(self, image, ic_index, k_index):
        self.image_shape = tuple(image.shape)
        ic = self.in_channels[ic_index]
        groups = ic
        k = self.kernel_sizes[k_index]
        padding = self.padding_config[k]
        weight = self._slice_weight(
            self.weight, ic // groups, ic, k)
        return F.conv2d(
            image, weight, None, self.stride, padding, self.dilation, groups)

    def get_num_parameters_macs(self, ic_index, k_index):
        _, c, h, w = self.image_shape
        ic = self.in_channels[ic_index]
        k = self.kernel_sizes[k_index]
        params = k * k * ic
        macs = k * k * ic * (h / self.stride) * (w / self.stride)
        macs = int(macs)
        return params, macs


class PointwiseConv2dSame(Conv2dSame):
    def __init__(
            self, in_channels, out_channels,
            stride=1, dilation=1, groups=1,
            bias=True):
        super(PointwiseConv2dSame, self).__init__(
            in_channels, out_channels, kernel_sizes=[1],
            stride=stride, dilation=dilation, groups=groups)

    def forward(self, image, ic_index, oc_index):
        self.image_shape = tuple(image.shape)
        ic = self.in_channels[ic_index]
        oc = self.out_channels[oc_index]
        padding = self.padding_config[1]
        weight = self._slice_weight(
            self.weight, ic, oc, 1)
        return F.conv2d(image, weight, None, self.stride, padding, self.dilation, 1)

    def get_num_parameters_macs(self, ic_index, oc_index):
        _, c, h, w = self.image_shape
        ic = self.in_channels[ic_index]
        oc = self.out_channels[oc_index]
        params = ic * oc
        macs = ic * oc * (h / self.stride) * (w / self.stride)
        macs = int(macs)
        return params, macs


class Activation(nn.Module):
    mapping = {
        'elu': nn.ELU,
        'selu': nn.SELU,
        'celu': nn.CELU,
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'leakyrelu': nn.LeakyReLU,
        'prelu': nn.PReLU,
        'rrelu': nn.RReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'hswish': h_swish,
        'hsigmoid': h_sigmoid,
    }
    macs_mapping = {
        'elu': lambda x: 2 * x,
        'selu': lambda x: 3 * x,
        'celu': lambda x: 3 * x,
        'relu': lambda x: 0,
        'relu6': lambda x: 0,
        'leakyrelu': lambda x: x,
        'prelu': lambda x: x,
        'rrelu': lambda x: x,
        'sigmoid': lambda x: 2 * x,
        'tanh': lambda x: 4 * x,
        'hswish': lambda x: 2 * x,
        'hsigmoid': lambda x: x,
    }

    def __init__(self, activation_fns):
        super(Activation, self).__init__()
        self.activations = [self.mapping[fn]() for fn in activation_fns]
        self.activations_names = activation_fns

    def forward(self, image, act_index):
        self.image_shape = tuple(image.shape)
        return self.activations[act_index](image)

    def get_num_parameters_macs(self, act_index):
        act_name = self.activations_names[act_index]
        num_pixels = np.prod(np.array(self.image_shape))
        macs = self.macs_mapping[act_name](num_pixels)
        return 0, macs



class SEModule(nn.Module):
    REDUCTION = 4
    def __init__(self, channel):
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION

        num_mid = _make_divisible(self.channel // self.reduction, divisor=8)
        self.num_mid = num_mid

        self.fc = nn.Sequential(OrderedDict([
            ('reduce', nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('expand', nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
            ('h_sigmoid', Hsigmoid(inplace=True)),
        ]))

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y


class DynamicSE(SEModule):
    def __init__(self, channels):
        max_channel = max(channels)
        super(DynamicSE, self).__init__(max_channel)

    def forward(self, x):
        self.image_shape = tuple(x.shape)
        in_channel = x.size(1)
        num_mid = _make_divisible(in_channel // self.reduction, divisor=8)
        self.in_channel, self.num_mid = in_channel, num_mid
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        # reduce
        reduce_conv = self.fc.reduce
        reduce_filter = reduce_conv.weight[:num_mid, :in_channel, :, :].contiguous()
        reduce_bias = reduce_conv.bias[:num_mid] if reduce_conv.bias is not None else None
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.relu(y)
        # expand
        expand_conv = self.fc.expand
        expand_filter = expand_conv.weight[:in_channel, :num_mid, :, :].contiguous()
        expand_bias = expand_conv.bias[:in_channel] if expand_conv.bias is not None else None
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.h_sigmoid(y)
        return x * y

    def get_num_parameters_macs(self):
        _, c, h, w = self.image_shape
        params = self.in_channel * self.num_mid * 2
        macs = self.in_channel * self.num_mid * 2 * h * w
        return params, macs

