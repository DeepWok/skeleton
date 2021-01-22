import functools

from torch import nn

from .common import (
    Sequential, BatchNorm2d, Conv2dSame, AdaptiveIdentity,
    ChannelShuffle, NoConnection)
from .base import ModelBase

class MNasConv(nn.Module):
    def __init__(
            self, inputs=None, outputs=None, stride=None, affine=True,
            kernel=None, expansion=1, groups=1, **kwargs):
        super().__init__()
        expanded = outputs * expansion
        layers = [
            # 1 * 1 group conv with expansion
            Conv2dSame(inputs, expanded, 1, 1, groups=groups, bias=False),
        ]
        layers += [
            # DW conv
            Conv2dSame(
                expanded, expanded, kernel, stride,
                groups=expanded, bias=False),
            nn.ReLU(inplace=True),
        ]
        layers += [
            # 1 * 1 group conv
            Conv2dSame(expanded, outputs, 1, 1, groups=groups, bias=False),
            BatchNorm2d(outputs, affine=affine),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MNasNoConnection(NoConnection):
    def __init__(self, inputs=None, outputs=None, stride=None, affine=True):
        super().__init__(inputs, outputs, stride)


TYPES = {
    '''
    'none': MNasNoConnection,
    '''
    'k3_e1': functools.partial(MNasConv, kernel=3, group=1, expansion=1),
    'k3_e1_g2': functools.partial(MNasConv, kernel=3, group=2, expansion=1),
    'k3_e3': functools.partial(MNasConv, kernel=3, group=1, expansion=3),
    'k3_e6': functools.partial(MNasConv, kernel=3, group=1, expansion=6),
    'k5_e1': functools.partial(MNasConv, kernel=5, group=1, expansion=1),
    'k5_e1_g2': functools.partial(MNasConv, kernel=5, group=2, expansion=1),
    'k5_e3': functools.partial(MNasConv, kernel=5, group=1, expansion=3),
    'k5_e6': functools.partial(MNasConv, kernel=5, group=1, expansion=6),
}


class MNasMixture(nn.Module):
    dropout_rate = 0.2
    def __init__(
            self, mixture_type, inputs, outputs, stride):
        super().__init__()
        self.mixture = mixture_type(
            TYPES, inputs, outputs, stride)
        if inputs == outputs and stride == 1:
            self.shortcut = True
        else:
            self.shortcut = False
        self.shortcut_connection = AdaptiveIdentity(inputs, outputs, stride, norm=True, relu=False)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(
            self, x, i=None, scale=None,
            search_mode=False, model_detach=False):
        m = self.mixture(x, i, scale, search_mode=search_mode, model_detach=model_detach)
        return self.dropout(m + self.shortcut_connection(x))
        # if self.shortcut:
        #     return m + x
        # else:
        #     return m
        '''
        return m + self.shortcut_connection(x)
        '''


class MNasNet(ModelBase):
    name = 'mnasnet'
    ops_dict = TYPES
    first_filter = 32
    final_filter = 512
    dropout_rate = 0.6
    filters = [40, 64, 96, 128, 192, 320]
    repeats = [2, 6, 6, 6, 6, 6]
    repeats = [2, 3, 3, 3, 3, 3]
    # this includes the first stride
    strides = [1, 2, 2, 1, 1, 2, 1]

    #for cifar10
    filters = [64, 96, 128, 320]
    repeats = [2, 3, 3, 3]
    repeats = [2, 5, 5, 5]
    # this includes the first stride
    strides = [1, 2, 2, 2, 1]


    def __init__(self, num_classes, mixture_type):
        super().__init__(num_classes, mixture_type)
        inputs = self.first_filter
        first_stride, *strides = self.strides
        self.conv0 = Sequential(
            Conv2dSame(3, inputs, 3, first_stride, bias=False),
            BatchNorm2d(inputs),
            nn.Dropout(self.dropout_rate),
        )
        iterer = zip(self.filters, strides, self.repeats)
        outputs = None
        layer_index = 0
        for outputs, stride, repeat in iterer:
            for _ in range(repeat):
                mixture = MNasMixture(
                    mixture_type, inputs, outputs, stride)
                setattr(self, f'layer_{layer_index}', mixture)
                layer_index += 1
                inputs = outputs
                stride = 1
        self.num_layers = layer_index
        # classifier
        self.outconv = Sequential(
            nn.ReLU(inplace=True),
            Conv2dSame(outputs, self.final_filter, 1, bias=False),
            BatchNorm2d(self.final_filter),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.final_filter, num_classes)

    def forward(
            self, x, ops, ops_scales, search_mode=False, model_detach=False):
        conved = self.conv0(x)
        for i in range(self.num_layers):
            op_index = ops[i]
            op_scale = ops_scales[i][op_index]
            conv_cls = getattr(self, f'layer_{i}')
            conved = conv_cls(
                conved, op_index, op_scale,
                search_mode=search_mode, model_detach=model_detach)
        output = self.outconv(conved)
        pooled = self.pool(output)
        return self.fc(pooled.squeeze(-1).squeeze(-1))

