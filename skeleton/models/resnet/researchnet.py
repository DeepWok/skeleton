"""
ResearchNet
Search using ResNet as a developing backbone
"""

import math
import functools
import torch

import torch.nn.functional as F
from torch import nn
from flareon.models.base import ModelBase
from ..base import ModelBase
from ..common import(
    BatchNorm2d, AdaptiveIdentity, ChannelShuffle)
from ..modules import (
    Conv2dSame, DepthwiseConv2dSame, PointwiseConv2dSame,
    DynamicBatchNorm2d, Activation, DynamicSE)


__all__ = ['ResearchNet110']


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class MixtureBasicBlock(nn.Module):
    block_expansion = 1
    def __init__(
            self, in_planes, planes, stride=1,
            kernel_sizes=[1], expansions=[1], activations=["relu"],
            option='A', dropout=False):
        super().__init__()
        self.dropout = dropout
        expanded = [in_planes * e for e in expansions]
        self.conv1 = Conv2dSame(
            [in_planes], expanded,
            kernel_sizes=kernel_sizes, stride=stride)
        self.bn1 = DynamicBatchNorm2d(expanded)
        self.act1 = Activation(activations)
        self.drop1 = nn.Dropout(p=0.3)
        self.conv2 = Conv2dSame(
            expanded, [planes],
            kernel_sizes=kernel_sizes, stride=1)
        self.bn2 = DynamicBatchNorm2d([planes])
        self.act2 = Activation(activations)
        self.drop2 = nn.Dropout(p=0.3)

        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x:
                    F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, planes,
                        kernel_size=1,
                        stride=stride),
                    nn.BatchNorm2d(planes))

    def forward(self, x, indexes_dict, scales_dict, scales_enable=True):
        exps = indexes_dict['exps']
        conv0_k = indexes_dict['conv0_k']
        conv1_k = indexes_dict['conv1_k']
        conv0_act = indexes_dict['conv0_act']
        conv1_act = indexes_dict['conv1_act']
        residual = x
        out = self.conv1(x, 0, exps, conv0_k)
        if scales_enable:
            out = out * scales_dict['exps'][exps] * scales_dict["conv0_k"][conv0_k]
        out = self.bn1(out)
        out = self.act1(out, conv0_act)
        if self.dropout:
            out = self.drop1(out)
        if scales_enable:
            out = out * scales_dict['conv0_act'][conv0_act]
        out = self.conv2(out, exps, 0, conv1_k)
        if scales_enable:
            out = out * scales_dict["conv1_k"][conv1_k]
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.act2(out, conv1_act)
        if scales_enable:
            out = out * scales_dict["conv1_act"][conv1_act]
        return out

    def get_num_parameters_macs(self, scales_dict, indexes_dict):
        exps = indexes_dict['exps']
        conv0_k = indexes_dict['conv0_k']
        conv1_k = indexes_dict['conv1_k']
        conv0_act = indexes_dict['conv0_act']
        conv1_act = indexes_dict['conv1_act']

        params_conv1, macs_conv1 = self.conv1.get_num_parameters_macs(
            0, exps, conv0_k)
        params_bn1, macs_bn1 = self.bn1.get_num_parameters_macs()
        params_act1, macs_act1 = self.act1.get_num_parameters_macs(conv0_act)

        params_conv2, macs_conv2 = self.conv2.get_num_parameters_macs(
            exps, 0, conv1_k)
        params_bn2, macs_bn2 = self.bn2.get_num_parameters_macs()
        params_act2, macs_act2 = self.act2.get_num_parameters_macs(conv1_act)

        params = params_conv1 + params_bn1 + params_act1 + params_conv2 + params_bn2 + params_act2
        macs = macs_conv1 + macs_bn1 + macs_act1 + macs_conv2 + macs_bn2 + macs_act2
        return params, macs


class ResearchNetBase(ModelBase):
    def __init__(
            self, block, num_blocks,
            num_classes=10, search_space=None, *args, **kwargs):
        super(ResearchNetBase, self).__init__(num_classes, *args, **kwargs)
        self.in_planes = 16
        self.search_space = search_space
        self.search_space_init = {
            "expansions": search_space["layer0_block0_conv0_ks"],
            "kernel_sizes": search_space["layer0_block0_conv0_ks"],
            "activations": search_space["layer0_block0_conv0_acts"],
        }
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(
            block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.linear_macs =  64 * num_classes
        self._initialize_weights()
        # self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_planes, planes, stride, **self.search_space_init))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, names, indexes, posteriors, search_mode, scales_enable=True):
        posteriors_dict, indexes_dict = self.process_posteriors(names, posteriors, indexes)
        self.conv1_image_shape = x.shape
        out = F.relu(self.bn1(self.conv1(x)))
        layers = [self.layer1, self.layer2, self.layer3]
        for i, l in enumerate(layers):
            for j, b in enumerate(l):
                block_dict = {
                    'exps': indexes_dict[f"layer{i}_block{j}_exps"],
                    'conv0_k': indexes_dict[f"layer{i}_block{j}_conv0_ks"],
                    'conv1_k': indexes_dict[f"layer{i}_block{j}_conv1_ks"],
                    'conv0_act': indexes_dict[f"layer{i}_block{j}_conv0_acts"],
                    'conv1_act': indexes_dict[f"layer{i}_block{j}_conv1_acts"]}
                block_scales_dict = {
                    'exps': posteriors_dict[f"layer{i}_block{j}_exps"],
                    'conv0_k': posteriors_dict[f"layer{i}_block{j}_conv0_ks"],
                    'conv1_k': posteriors_dict[f"layer{i}_block{j}_conv1_ks"],
                    'conv0_act': posteriors_dict[f"layer{i}_block{j}_conv0_acts"],
                    'conv1_act': posteriors_dict[f"layer{i}_block{j}_conv1_acts"]}
                out = b(out, block_dict, block_scales_dict, scales_enable=scales_enable)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_num_params_macs(self, names, indexes, posteriors):
        posteriors_dict, indexes_dict = self.process_posteriors(names, posteriors, indexes)
        b, c, h, w = self.conv1_image_shape
        params, macs = 3*3*3*16, c*h*w*3*3
        # out = F.relu(self.bn1(self.conv1(x)))
        layers = [self.layer1, self.layer2, self.layer3]
        for i, l in enumerate(layers):
            for j, b in enumerate(l):
                block_dict = {
                    'exps': indexes_dict[f"layer{i}_block{j}_exps"],
                    'conv0_k': indexes_dict[f"layer{i}_block{j}_conv0_ks"],
                    'conv1_k': indexes_dict[f"layer{i}_block{j}_conv1_ks"],
                    'conv0_act': indexes_dict[f"layer{i}_block{j}_conv0_acts"],
                    'conv1_act': indexes_dict[f"layer{i}_block{j}_conv1_acts"]}
                block_scales_dict = {
                    'exps': posteriors_dict[f"layer{i}_block{j}_exps"],
                    'conv0_k': posteriors_dict[f"layer{i}_block{j}_conv0_ks"],
                    'conv1_k': posteriors_dict[f"layer{i}_block{j}_conv1_ks"],
                    'conv0_act': posteriors_dict[f"layer{i}_block{j}_conv0_acts"],
                    'conv1_act': posteriors_dict[f"layer{i}_block{j}_conv1_acts"]}
                p, m = b.get_num_parameters_macs(
                    block_dict, block_scales_dict)
                params, macs = params + p, macs + m
        params, macs = params + self.linear_macs, macs + self.linear_macs
        return params, macs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2dSame):
                ksize = max(m.kernel_sizes)
                oc = max(m.out_channels)
                n = ksize * ksize * oc
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


class ResearchNet56(ResearchNetBase):
    def __init__(
            self, num_classes=10, search_space=None, cfgs=None, first_stride=None):
        super(ResearchNet56, self).__init__(
            block=MixtureBasicBlock, num_blocks=[9, 9, 9],
            num_classes=num_classes, search_space=search_space)


class WideResearchNet56(ResearchNetBase):
    factor = 2
    def __init__(
            self, num_classes=10, search_space=None, cfgs=None, first_stride=None):
        super(WideResearchNet56, self).__init__(
            block=MixtureBasicBlock, num_blocks=[9, 9, 9],
            num_classes=num_classes, search_space=search_space)
        block = MixtureBasicBlock
        num_blocks=[9, 9, 9]

        self.in_planes = 32 * self.factor
        self.search_space = search_space
        self.search_space_init = {
            "expansions": search_space["layer0_block0_conv0_ks"],
            "kernel_sizes": search_space["layer0_block0_conv0_ks"],
            "activations": search_space["layer0_block0_conv0_acts"],
        }
        self.conv1 = nn.Conv2d(
            3, 32 * self.factor, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32 * self.factor)
        self.layer1 = self._make_layer(
            block, 32 * self.factor, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            block, 48 * self.factor, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            block, 64 * self.factor, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * self.factor, num_classes)
        self.linear_macs =  64 * num_classes * self.factor
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_planes, planes, stride,
                option='B',
                dropout=True,
                **self.search_space_init))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def get_num_params_macs(self, names, indexes, posteriors):
        posteriors_dict, indexes_dict = self.process_posteriors(names, posteriors, indexes)
        b, c, h, w = self.conv1_image_shape
        params, macs = 3*3*3*32*self.factor, c*h*w*3*3
        # out = F.relu(self.bn1(self.conv1(x)))
        layers = [self.layer1, self.layer2, self.layer3]
        for i, l in enumerate(layers):
            for j, b in enumerate(l):
                block_dict = {
                    'exps': indexes_dict[f"layer{i}_block{j}_exps"],
                    'conv0_k': indexes_dict[f"layer{i}_block{j}_conv0_ks"],
                    'conv1_k': indexes_dict[f"layer{i}_block{j}_conv1_ks"],
                    'conv0_act': indexes_dict[f"layer{i}_block{j}_conv0_acts"],
                    'conv1_act': indexes_dict[f"layer{i}_block{j}_conv1_acts"]}
                block_scales_dict = {
                    'exps': posteriors_dict[f"layer{i}_block{j}_exps"],
                    'conv0_k': posteriors_dict[f"layer{i}_block{j}_conv0_ks"],
                    'conv1_k': posteriors_dict[f"layer{i}_block{j}_conv1_ks"],
                    'conv0_act': posteriors_dict[f"layer{i}_block{j}_conv0_acts"],
                    'conv1_act': posteriors_dict[f"layer{i}_block{j}_conv1_acts"]}
                p, m = b.get_num_parameters_macs(
                    block_dict, block_scales_dict)
                params, macs = params + p, macs + m
        params, macs = params + self.linear_macs, macs + self.linear_macs
        return params, macs


class SmallResearchNet56(ResearchNetBase):
    def __init__(
            self, num_classes=10, search_space=None, cfgs=None, first_stride=None):
        super(SmallResearchNet56, self).__init__(
            block=MixtureBasicBlock, num_blocks=[5, 5, 5],
            num_classes=num_classes, search_space=search_space)
