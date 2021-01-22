import torch

from .base import ModelBase


class LeNet5(ModelBase):
    name = 'lenet5'
    input_size = (1, 28, 28)

    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(self.input_size[0], 6, 5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16 * 4 * 4, 120),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
