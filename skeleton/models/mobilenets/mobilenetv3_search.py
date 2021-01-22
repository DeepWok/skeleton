import torch
import math
from torch import nn
from flareon.models.base import ModelBase
from .util import (
    _make_divisible, h_swish, h_sigmoid, conv_1x1_bn, conv_3x3_bn,
    InvertedResidual)
from .mixture import MBConv
from ..modules import Conv2dSame


class MobileNetV3Search(ModelBase):
    default_cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    mode = 'large'
    width_mult = 1.
    def __init__(
            self, num_classes=1000, search_space=None, cfgs=None,
            first_stride=1, *args, **kwargs):
        super().__init__(num_classes, *args, **kwargs)
        # setting of inverted residual blocks
        mode = self.mode
        self.search_space = search_space
        self.cfgs = self.default_cfgs if cfgs is None else cfgs
        self.num_layers = len(self.cfgs)

        width_mult = self.width_mult
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        conv0_k = search_space['conv0_k']
        self.conv0 = Conv2dSame(
            [3], [input_channel], kernel_sizes=conv0_k,
            stride=first_stride)
        self.conv0_act = h_swish()
        # building inverted residual blocks
        blocks = nn.ModuleList()
        block = MBConv
        for i, meta in enumerate(self.cfgs):
            _, exp, c, se, hs, s = meta
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(exp * width_mult, 8)
            depthwise_ks = search_space[f'mbconv{i}_depthwise_ks']
            depthwise_acts = search_space[f'mbconv{i}_depthwise_acts']
            expansions = search_space[f'mbconv{i}_exps']
            pointwise_acts = search_space[f'mbconv{i}_pointwise_acts']
            tmp = block(
                input_channel, output_channel,
                stride=s, use_se=se,
                depthwise_ks=depthwise_ks,
                depthwise_acts=depthwise_acts,
                expansions=expansions,
                pointwise_acts=pointwise_acts)
            blocks.append(tmp)
            input_channel = output_channel
        self.blocks = blocks
        # building last several layers
        self.final_conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )
        self.classifier_params_macs = exp_size * output_channel + output_channel * num_classes
        self._initialize_weights()

    # def process_posteriors(self, names, posteriors, indexes):
    #     if isinstance(posteriors, list):
    #         posteriors_dict = dict([
    #             (n, p[~torch.isnan(p)])
    #             for n, p in zip(names, posteriors)])
    #         indexes_dict = dict([(n, i) for n, i in zip(names, indexes)])
    #     else:
    #         posteriors = torch.transpose(posteriors, 1, 2)
    #         posteriors_dict = dict([
    #             (n, p[~torch.isnan(p)])
    #             for n, p in
    #             # zip(names, posteriors)])
    #             zip(names, posteriors[0])])
    #         indexes_dict = dict([(n, i) for n, i in zip(names, indexes[0])])
    #     return posteriors_dict, indexes_dict

    def forward(self, x, names, indexes, posteriors, search_mode, scales_enable=True):
        posteriors_dict, indexes_dict = self.process_posteriors(names, posteriors, indexes)
        self.first_conv_image_shape = x.shape
        x = self.conv0_act(
            self.conv0(x, 0, 0, indexes_dict['conv0_k']))
        for i, block in enumerate(self.blocks):
            block_dict = {
                'depthwise_ks': indexes_dict[f"mbconv{i}_depthwise_ks"],
                'depthwise_acts':indexes_dict[f"mbconv{i}_depthwise_acts"],
                'exps': indexes_dict[f"mbconv{i}_exps"],
                'pointwise_acts': indexes_dict[f"mbconv{i}_pointwise_acts"],
            }
            block_scale_dict = {
                'depthwise_ks': posteriors_dict[f"mbconv{i}_depthwise_ks"],
                'depthwise_acts':posteriors_dict[f"mbconv{i}_depthwise_acts"],
                'exps': posteriors_dict[f"mbconv{i}_exps"],
                'pointwise_acts': posteriors_dict[f"mbconv{i}_pointwise_acts"],
            }
            x = block(x, block_dict, block_scale_dict, scales_enable=scales_enable)
        self.final_conv_image_shape = x.shape
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        del posteriors_dict, indexes_dict
        return x

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

    def get_num_params_macs(self, names, posteriors, indexes):
        posteriors_dict, indexes_dict = self.process_posteriors(
            names, posteriors, indexes)
        n, c, h, w = self.first_conv_image_shape
        params, macs = 0, c * h * w * 2
        p, m = self.conv0.get_num_parameters_macs(0, 0, indexes_dict['conv0_k'])
        params, macs = params + p, macs + m
        for i, block in enumerate(self.blocks):
            block_dict = {
                'depthwise_ks': indexes_dict[f"mbconv{i}_depthwise_ks"],
                'depthwise_acts':indexes_dict[f"mbconv{i}_depthwise_acts"],
                'exps': indexes_dict[f"mbconv{i}_exps"],
                'pointwise_acts': indexes_dict[f"mbconv{i}_pointwise_acts"],
            }
            block_scale_dict = {
                'depthwise_ks': posteriors_dict[f"mbconv{i}_depthwise_ks"],
                'depthwise_acts':posteriors_dict[f"mbconv{i}_depthwise_acts"],
                'exps': posteriors_dict[f"mbconv{i}_exps"],
                'pointwise_acts': posteriors_dict[f"mbconv{i}_pointwise_acts"],
            }
            p, m = block.get_num_parameters_macs(block_dict, block_scale_dict)
            params, macs = params + p, macs + m
        params += self.classifier_params_macs
        macs += self.classifier_params_macs
        return params, macs
        # x = self.classifier(x)


class MobileNetV3SearchSmall(MobileNetV3Search):
    mode = 'small'
