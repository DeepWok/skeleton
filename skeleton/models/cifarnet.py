from torch import nn

from .common import Sequential, BatchNorm2d, Conv2dSame
from .base import ModelBase


class CifarNet(ModelBase):
    name = 'cifarnet'
    filters = [64, 64, 128, 128, 128, 192, 192, 192]
    kernels = [3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 2, 1, 1, 2, 1, 1]
    dropout = [False, False, False, True, False, False, True, False]

    def __init__(self, num_classes):
        super().__init__(num_classes)
        inputs = 3
        iterer = zip(self.kernels, self.filters, self.strides, self.dropout)
        outputs = None
        layers = []
        for k, outputs, stride, dropout in iterer:
            layers += [
                Conv2dSame(inputs, outputs, k, stride, bias=False),
                BatchNorm2d(outputs),
                nn.ReLU(inplace=False),
            ]
            if dropout:
                layers.append(nn.Dropout(p=0.5))
            inputs = outputs
        self.layers = Sequential(*layers)
        # classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(outputs, num_classes)

    def forward(self, x):
        extracted = self.layers(x)
        if isinstance(extracted, tuple):
            extracted, _ = extracted
        pooled = self.pool(extracted)
        return self.fc(pooled.squeeze(-1).squeeze(-1))
