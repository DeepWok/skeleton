import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import ModelBase
from ..common import(
    BatchNorm2d, AdaptiveIdentity, ChannelShuffle)
from ..modules import (
    Conv2dSame, DepthwiseConv2dSame, PointwiseConv2dSame,
    DynamicBatchNorm2d, Activation, DynamicSE)



class BasicBlock(nn.Module):
    def __init__(
            self, in_planes, out_planes, stride, dropRate=0.0,
            kernel_sizes_conv0=[3], kernel_sizes_conv1=[3],
            expansions=[1], activations_conv0=["relu"], activations_conv1=["relu"]):
        super(BasicBlock, self).__init__()
        expanded = [int(out_planes * e) for e in expansions]

        self.bn0 = nn.BatchNorm2d(in_planes)
        self.act0 = Activation(activations_conv0)
        self.conv0 = Conv2dSame(
            [in_planes], expanded,
            kernel_sizes=kernel_sizes_conv0, stride=stride, bias=False)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)

        self.bn1 = DynamicBatchNorm2d(expanded)
        self.act1 = Activation(activations_conv1)
        self.conv1 = Conv2dSame(
            expanded, [out_planes],
            kernel_sizes=kernel_sizes_conv1, stride=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_planes)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
        #                        padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x, indexes_dict, scales_dict, scales_enable=True):
        exps = indexes_dict['exps']
        conv0_k = indexes_dict['conv0_k']
        conv1_k = indexes_dict['conv1_k']
        conv0_act = indexes_dict['conv0_act']
        conv1_act = indexes_dict['conv1_act']

        conv0_act_scale = scales_dict['conv0_act'][conv0_act]
        conv1_act_scale = scales_dict['conv0_act'][conv1_act]
        x = self.act0(self.bn0(x), conv0_act)
        if scales_enable:
            x = x * conv0_act_scale
        conv0 = self.conv0(
            x, 0, exps, conv0_k)
        if scales_enable:
            conv0 = conv0 * scales_dict['exps'][exps] * scales_dict["conv0_k"][conv0_k]
        out = self.act1(self.bn1(conv0), conv1_act)
        if scales_enable:
            out = out * conv1_act_scale
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv1(out, exps, 0, conv1_k)
        if scales_enable:
            out = out * scales_dict["conv1_k"][conv0_k]
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, search_space_init=None):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, search_space_init)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, search_space_init):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1, dropRate,
                    **search_space_init))
        return nn.ModuleList(layers)

    def forward(self, x, i, indexes_dict, scales_dict, scales_enable):
        for j, l in enumerate(self.layer):
            block_dict = {
                'exps': indexes_dict[f"layer{i}_block{j}_exps"],
                'conv0_k': indexes_dict[f"layer{i}_block{j}_conv0_ks"],
                'conv1_k': indexes_dict[f"layer{i}_block{j}_conv1_ks"],
                'conv0_act': indexes_dict[f"layer{i}_block{j}_conv0_acts"],
                'conv1_act': indexes_dict[f"layer{i}_block{j}_conv1_acts"]}
            block_scales_dict = {
                'exps': scales_dict[f"layer{i}_block{j}_exps"],
                'conv0_k': scales_dict[f"layer{i}_block{j}_conv0_ks"],
                'conv1_k': scales_dict[f"layer{i}_block{j}_conv1_ks"],
                'conv0_act': scales_dict[f"layer{i}_block{j}_conv0_acts"],
                'conv1_act': scales_dict[f"layer{i}_block{j}_conv1_acts"]
            }
            x = l(x, block_dict, block_scales_dict, scales_enable)
        return x


class WideResearchNet40(ModelBase):
    def __init__(
            self, num_classes=10, search_space=None, cfgs=None, first_stride=None,
            depth=40, widen_factor=4, dropRate=0.0, use_FNandWN=False, *args, **kwargs):
        super(WideResearchNet40, self).__init__(num_classes, *args, **kwargs)
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.search_space = search_space
        self.search_space_init = {
            "expansions": search_space["layer0_block0_exps"],
            "kernel_sizes_conv0": search_space["layer0_block0_conv0_ks"],
            "kernel_sizes_conv1": search_space["layer0_block0_conv1_ks"],
            "activations_conv0": search_space["layer0_block0_conv0_acts"],
            "activations_conv1": search_space["layer0_block0_conv1_acts"],
        }
        print(self.search_space_init)
        self.use_FNandWN = use_FNandWN
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1,
            padding=1, bias=False)
        # 1st block
        self.block0 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, self.search_space_init)
        # # 1st sub-block
        # self.sub_block0 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, self.search_space_init)
        # 2nd block
        self.block1 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, self.search_space_init)
        # 3rd block
        self.block2 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, self.search_space_init)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        if self.use_FNandWN:
            self.fc = nn.Linear(nChannels[3], num_classes, bias = False)
        else:
            self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not self.use_FNandWN:
                m.bias.data.zero_()

    def forward(self, x, names, indexes, posteriors, search_mode, scales_enable=True):
        # print(torch.sum(torch.stack(indexes)), scales_enable)
        posteriors_dict, indexes_dict = self.process_posteriors(names, posteriors, indexes)
        out = self.conv1(x)
        out = self.block0(out, 0, indexes_dict, posteriors_dict, scales_enable)
        out = self.block1(out, 1, indexes_dict, posteriors_dict, scales_enable)
        out = self.block2(out, 2, indexes_dict, posteriors_dict, scales_enable)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if self.use_FNandWN:
            out = F.normalize(out, p=2, dim=1)
            for _, module in self.fc.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = F.normalize(module.weight, p=2, dim=1)
        return self.fc(out)

    def get_num_params_macs(self, names, indexes, posteriors):
        # forget this for now
        return 0, 0
        # posteriors_dict, indexes_dict = self.process_posteriors(names, posteriors, indexes)
        # b, c, h, w = self.conv1_image_shape
        # params, macs = 3*3*3*16, c*h*w*3*3
        # # out = F.relu(self.bn1(self.conv1(x)))
        # layers = [self.layer1, self.layer2, self.layer3]
        # for i, l in enumerate(layers):
        #     for j, b in enumerate(l):
        #         block_dict = {
        #             'exps': indexes_dict[f"layer{i}_block{j}_exps"],
        #             'conv0_k': indexes_dict[f"layer{i}_block{j}_conv0_ks"],
        #             'conv1_k': indexes_dict[f"layer{i}_block{j}_conv1_ks"],
        #             'conv0_act': indexes_dict[f"layer{i}_block{j}_conv0_acts"],
        #             'conv1_act': indexes_dict[f"layer{i}_block{j}_conv1_acts"]}
        #         block_scales_dict = {
        #             'exps': posteriors_dict[f"layer{i}_block{j}_exps"],
        #             'conv0_k': posteriors_dict[f"layer{i}_block{j}_conv0_ks"],
        #             'conv1_k': posteriors_dict[f"layer{i}_block{j}_conv1_ks"],
        #             'conv0_act': posteriors_dict[f"layer{i}_block{j}_conv0_acts"],
        #             'conv1_act': posteriors_dict[f"layer{i}_block{j}_conv1_acts"]}
        #         p, m = b.get_num_parameters_macs(
        #             block_dict, block_scales_dict)
        #         params, macs = params + p, macs + m
        # params, macs = params + self.linear_macs, macs + self.linear_macs
        # return params, macs


class WideResearchNet28(WideResearchNet40):
    def __init__(
            self, num_classes=10, search_space=None, cfgs=None, first_stride=None,
            depth=28, widen_factor=10, dropRate=0.0, use_FNandWN=False, *args, **kwargs):
        super(WideResearchNet28, self).__init__(
            num_classes, search_space, cfgs, 
            first_stride, depth, widen_factor, dropRate, use_FNandWN, *args, **kwargs)
