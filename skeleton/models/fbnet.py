import functools

from torch import nn

from .common import (
    Sequential, BatchNorm2d, Conv2dSame, AdaptiveIdentity,
    ChannelShuffle, NoConnection)
from .base import ModelBase


class FBConv(nn.Module):
    def __init__(
            self, inputs=None, outputs=None, stride=None, affine=True,
            kernel=None, expansion=1, groups=1, **kwargs):
        super().__init__()
        expanded = outputs * expansion
        layers = [
            # 1 * 1 group conv with expansion
            Conv2dSame(inputs, expanded, 1, 1, groups=groups, bias=False),
            BatchNorm2d(expanded),
            nn.ReLU(inplace=True),
        ]
        if groups > 1:
            layers.append(ChannelShuffle(groups))
        layers += [
            # DW conv
            Conv2dSame(
                expanded, expanded, kernel, stride,
                groups=expanded, bias=False),
            BatchNorm2d(expanded),
            nn.ReLU(inplace=True),
        ]
        layers += [
            # 1 * 1 group conv
            Conv2dSame(expanded, outputs, 1, 1, groups=groups, bias=False),
            BatchNorm2d(outputs, affine=affine),
            # nn.ReLU(inplace=True),
        ]
        if groups > 1:
            layers.append(ChannelShuffle(groups))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FBNoConnection(NoConnection):
    def __init__(self, inputs=None, outputs=None, stride=None, affine=True):
        super().__init__(inputs, outputs, stride)


TYPES = {
    'none': FBNoConnection,
    'k3_e1': functools.partial(FBConv, kernel=3, group=1, expansion=1),
    'k3_e1_g2': functools.partial(FBConv, kernel=3, group=2, expansion=1),
    'k3_e3': functools.partial(FBConv, kernel=3, group=1, expansion=3),
    'k3_e6': functools.partial(FBConv, kernel=3, group=1, expansion=6),
    'k5_e1': functools.partial(FBConv, kernel=5, group=1, expansion=1),
    'k5_e1_g2': functools.partial(FBConv, kernel=5, group=2, expansion=1),
    'k5_e3': functools.partial(FBConv, kernel=5, group=1, expansion=3),
    'k5_e6': functools.partial(FBConv, kernel=5, group=1, expansion=6),
}


class FBMixture(nn.Module):
    def __init__(
            self, mixture_type, inputs, outputs, stride):
        super().__init__()
        self.mixture = mixture_type(
            TYPES, inputs, outputs, stride)
        self.shortcut = AdaptiveIdentity(
            inputs, outputs, stride, norm=True, relu=False)

    def forward(self, x, i=None, scale=None, eval_mode=False):
        m = self.mixture(x, i, scale, eval_mode)
        return m + self.shortcut(x)


class FBNet(ModelBase):
    name = 'fbnet'
    ops_dict = TYPES
    first_filter = 16
    final_filter = 1504
    filters = [16, 24, 32, 64, 112, 184, 352]
    repeats = [1, 4, 4, 4, 4, 4, 1]
    strides = {
        'large': [2, 1, 2, 2, 2, 1, 2, 1],
        'small': [1, 1, 2, 2, 1, 1, 2, 1],
    }

    def __init__(self, num_classes, mixture_type, size='small'):
        super().__init__(num_classes, mixture_type)
        inputs = self.first_filter
        first_stride, *strides = self.strides[size]
        self.conv0 = Sequential(
            Conv2dSame(3, inputs, 3, first_stride, bias=False),
            BatchNorm2d(inputs),
        )
        layers = []
        iterer = zip(self.filters, strides, self.repeats)
        outputs = None
        layer_index = 0
        for outputs, stride, repeat in iterer:
            for _ in range(repeat):
                mixture = FBMixture(
                    mixture_type, inputs, outputs, stride)
                setattr(self, f'layer_{layer_index}', mixture)
                layer_index += 1
                inputs = outputs
                stride = 1
        self.num_layers = layer_index
        # classifier
        self.outconv = Sequential(
            Conv2dSame(outputs, self.final_filter, 1, bias=False),
            BatchNorm2d(self.final_filter),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.final_filter, num_classes)

    def forward(self, x, ops, ops_scales, search_mode=False):
        conved = self.conv0(x)
        for i in range(self.num_layers):
            op_index = ops[i]
            op_scale = ops_scales[i][op_index]
            conv_cls = getattr(self, f'layer_{i}')
            conved = conv_cls(conved, op_index, op_scale, search_mode)
        output = self.outconv(conved)
        pooled = self.pool(output)
        return self.fc(pooled.squeeze(-1).squeeze(-1))


class FBNetSmall(FBNet):
    name = 'fbnetsmall'
    repeats = [1, 1, 1, 1, 1, 1, 1]
