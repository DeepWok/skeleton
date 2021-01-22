import functools

from torch import nn
from collections import OrderedDict
from .util import _make_divisible
from ..modules import (
    DepthwiseConv2dSame, PointwiseConv2dSame,
    DynamicBatchNorm2d, Activation, DynamicSE)



class MBConv(nn.Module):
    def __init__(
            self, inputs=None, outputs=None,
            stride=None, use_se=True,
            depthwise_ks=None, depthwise_acts=None,
            expansions=None, pointwise_acts=None,
            **kwargs):
        super(MBConv, self).__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inputs == outputs
        expanded = [inputs * i for i in expansions]
        # pw
        self.pointwise_expand = PointwiseConv2dSame(
            [inputs], expanded)
        self.pointwise_expand_bn = DynamicBatchNorm2d(expanded)
        self.pointwise_expand_act = Activation(pointwise_acts)
        # dw
        self.depthwise = DepthwiseConv2dSame(
            expanded,
            depthwise_ks,
            stride=stride)
        self.depthwise_bn = DynamicBatchNorm2d(expanded)
        self.depthwise_se = DynamicSE(expanded) if use_se else nn.Identity()
        self.depthwise_act = Activation(depthwise_acts)
        # pw
        self.pointwise = PointwiseConv2dSame(
            expanded,
            [outputs])
        self.pointwise_bn = DynamicBatchNorm2d([outputs])

    def forward(self, x, indexes_dict, scales_dict, scales_enable=True):
        self.image_shape = x.shape
        exps = indexes_dict['exps']
        pointwise_expand_act = indexes_dict['pointwise_acts']
        y = self.pointwise_expand(x, 0, exps)
        if scales_enable:
            y = y * scales_dict['exps'][exps]

        y = self.pointwise_expand_bn(y)
        y = self.pointwise_expand_act(y, pointwise_expand_act)
        if scales_enable:
            y = y * scales_dict['pointwise_acts'][pointwise_expand_act]

        depthwise_k = indexes_dict['depthwise_ks']
        depthwise_act = indexes_dict['depthwise_acts']
        y = self.depthwise(y, exps, depthwise_k)
        if scales_enable:
            y = y * scales_dict['depthwise_ks'][depthwise_k]

        y = self.depthwise_bn(y)
        y = self.depthwise_se(y)
        y = self.depthwise_act(y, depthwise_act)
        if scales_enable:
            y = y * scales_dict['depthwise_acts'][depthwise_act]

        y = self.pointwise(y, exps, 0)
        y = self.pointwise_bn(y)
        if self.identity:
            return x + y
        return y

    def get_num_parameters_macs(self, indexes_dict, scales_dict):
        exps = indexes_dict['exps']
        pointwise_expand_act = indexes_dict['pointwise_acts']

        params_pw_exp, macs_pw_exp = self.pointwise_expand.get_num_parameters_macs(0, exps)
        params_pw_bn_exp, macs_pw_bn_exp = \
            self.pointwise_expand_bn.get_num_parameters_macs()
        params_pw_act, macs_pw_act = \
            self.pointwise_expand_act.get_num_parameters_macs(
                pointwise_expand_act)

        depthwise_k = indexes_dict['depthwise_ks']
        depthwise_act = indexes_dict['depthwise_acts']

        params_dw, macs_dw = self.depthwise.get_num_parameters_macs(exps, depthwise_k)
        params_dw_bn, macs_dw_bn = self.depthwise_bn.get_num_parameters_macs()
        if isinstance(self.depthwise_se, nn.Identity):
            params_dw_se, macs_dw_se = 0, 0
        else:
            params_dw_se, macs_dw_se = self.depthwise_se.get_num_parameters_macs()
        params_dw_act, macs_dw_act = self.depthwise_act.get_num_parameters_macs(depthwise_act)

        params_pw, macs_pw = self.pointwise.get_num_parameters_macs(exps, 0)
        params_pw_bn, macs_pw_bn = self.pointwise_bn.get_num_parameters_macs()
        params = \
            params_pw_exp + params_pw_bn_exp + params_pw_act + params_dw + \
            params_dw_bn + params_dw_bn + params_dw_se + params_dw_act + \
            params_pw + params_pw_bn
        macs = \
            macs_pw_exp + macs_pw_bn_exp + macs_pw_act + macs_dw + \
            macs_dw_bn + macs_dw_bn + macs_dw_se + macs_dw_act + \
            macs_pw + macs_pw_bn
        # print(self.image_shape, params, macs)
        return params, macs

