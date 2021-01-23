import os
import math
import signal
import functools

import torch
import torchvision as tv

from ..utils import device
from .cutout import Cutout
from .autoaugment import ImageNetPolicy, CIFAR10Policy

torch.multiprocessing.set_sharing_strategy('file_system')


class AugmentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, augments, train):
        super().__init__()
        self.dataset = dataset
        self.augments = augments
        self.is_training = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        augment = self.augments['train' if self.is_training else 'eval']
        return augment(image), label


MOMENTS = {
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'svhn': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    'cifar10': ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    'cifar10plain': ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    'cifar100': ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
}
AUGMENT_POLICIES = {
    'svhn': {
        'eval': (
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['svhn'])
        ),
        'train': (
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['svhn'])
        )
    },
    'cifar10': {
        'eval': (
            tv.transforms.ToTensor(),
            # tv.transforms.Normalize(*MOMENTS['cifar10']),
        ),
        'train': (
            # tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            # tv.transforms.Normalize(*MOMENTS['cifar10']),
        ),
    },
    'cifar10plain': {
        'eval': (
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['cifar10plain']),
        ),
        'train': (
            # tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['cifar10plain']),
        ),
    },
    'cifar100': {
        'eval': (
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['cifar100']),
        ),
        'train': (
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['cifar100']),
        ),
    },
    'imagenet': {
        'eval': (
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['imagenet']),
        ),
        'train': (
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['imagenet']),
        ),
    },
    'mnist': {
        'eval': (
            tv.transforms.ToTensor(),
        ),
        'train': (
            tv.transforms.ToTensor(),
        ),
    },
}
INFO = {
    'mnist': {'num_classes': 10, 'shape': (1, 28, 28)},
    'svhn': {'num_classes': 10, 'shape': (3, 54, 54)},
    'cifar10': {'num_classes': 10, 'shape': (3, 32, 32)},
    'cifar10plain': {'num_classes': 10, 'shape': (3, 32, 32)},
    'cifar100': {'num_classes': 100, 'shape': (3, 32, 32)},
    'imagenet': {'num_classes': 1000, 'shape': (3, 224, 224)},
}


class DataLoader(torch.utils.data.DataLoader):
    pin_memory = True

    def __init__(self, dataset, augments, batch_size, shuffle, workers, info):
        augments = {k: tv.transforms.Compose(v) for k, v in augments.items()}
        dataset = AugmentDataset(dataset, augments, shuffle)
        super().__init__(
            dataset, batch_size, shuffle,
            pin_memory=self.pin_memory,
            num_workers=0, worker_init_fn=self.worker_init)
        self.num_classes = info['num_classes']
        self.shape = info['shape']

    @staticmethod
    def worker_init(x):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def train(self):
        self.dataset.is_training = True
        return self

    def eval(self):
        self.dataset.is_training = False
        return self

    def __iter__(self):
        for item in super().__iter__():
            yield [x.to(device) for x in item]


def Dataset(name, train, split=None, batch_size=128, workers=1):
    name = name.lower()
    if name == "cifar10plain":
        d_name = "cifar10"
    else:
        d_name = name
    if name == 'imagenet':
        path = f'./data/imagenet/{"train" if train else "val"}'
        dataset = tv.datasets.ImageFolder(path)
    else:
        cls = getattr(tv.datasets, d_name.upper())
        path = os.path.join('data', f'{d_name}')
        train_split = 'train' if train else 'test'
        if d_name == 'svhn':
            try:
                dataset = cls(path, split=train_split, download=False)
            except RuntimeError:
                dataset = cls(path, split=train_split, download=True)
        else:
            try:
                dataset = cls(path, train=train, download=False)
            except RuntimeError:
                dataset = cls(path, train=train, download=True)
    kwargs = {
        'augments': AUGMENT_POLICIES[name],
        'batch_size': batch_size,
        'shuffle': train,
        'info': INFO[name],
        'workers': workers,
    }
    Loader = functools.partial(DataLoader, **kwargs)
    if split is None:
        return Loader(dataset)
    points = len(dataset)
    split = math.floor(split * points)
    # enforce determinism right before random_split
    # to guarantee identical splits at each run
    # torch.manual_seed(0)
    train_set, search_set = torch.utils.data.dataset.random_split(
        dataset, [split, points - split])
    return Loader(train_set), Loader(search_set)
