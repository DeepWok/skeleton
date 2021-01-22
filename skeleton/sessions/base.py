import os
import torch
import functools
import multiprocessing
import pickle
import sklearn
import numpy as np
import yaml

from pprint import pformat
from collections import defaultdict

from ..pretty import log, AccuracyCounter, SummaryHook, TensorboardWriter 
from ..utils import (
        device, DataParallel,
        to_numpy, get_num_params, use_cuda, topk)
from ..datasets import Dataset, INFO
from ..models import factory


def _get(kv, key, desc):
    try:
        return kv[key.lower()]
    except KeyError:
        log.error_exit(
            f'Unknown {desc} name {key!r}, accepts: {", ".join(kv)}.')


class SessionBase:
    is_training = False

    def __init__(
            self, model_name, dataset_name,
            load_name=None, save_name=None,
            gpus=1, workers=None, batch_size=128,
            max_epochs=None, model_dir="models"):
        super().__init__()
        self.model_dir = model_dir
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.load_name = load_name
        self.save_name = save_name
        self.gpus = gpus
        self.max_epochs = max_epochs
        self.imgs_seen = 0
        num_classes = INFO[dataset_name]['num_classes']
        if workers is None:
            self.workers = multiprocessing.cpu_count()
        else:
            self.workers = workers
        self.batch_size = batch_size
        # dataset
        self.Dataset = functools.partial(
            Dataset, name=self.dataset_name,
            train=True,
            split=None,
            batch_size=self.batch_size, workers=self.workers)
        # dataset
        self.TestDataset = functools.partial(
            Dataset, name=self.dataset_name,
            train=False,
            split=None,
            batch_size=self.batch_size, workers=self.workers)

        self.train_dataset, self.test_dataset = self.Dataset(), self.TestDataset()
        # model
        state = {}
        self._info = state.pop('.info', None)
        model_cls = _get(factory, self.model_name, 'model')
        model = model_cls(num_classes=num_classes)

        if self.is_training and self.gpus > 1:
            log.info(
                f'Using {self.gpus} '
                f'device{"s" if self.gpus > 1 else ""}...')
            model = DataParallel(model, range(self.gpus))

        self.model = model.to(device)
        if self.load_name:
            self.load(state)
        self.tb_writer = TensorboardWriter(
            dataset_name,
            model_name)

    @property
    def prefix(self):
        return f'{self.model_name.lower()}.{self.dataset_name.lower()}'

    def load_path(self, suffix=None):
        suffix = f'.{suffix}' if suffix else ''
        prefix = self.prefix
        return f'{self.model_dir}/{prefix}.{self.load_name}{suffix}.pth'

    def save_path(self, suffix=None):
        suffix = f'.{suffix}' if suffix else ''
        return f'{self.model_dir}/{self.prefix}.{self.save_name}{suffix}.pth'

    def load_state(self, suffix_or_path_or_state=None):
        s = suffix_or_path_or_state
        if s is not None and not isinstance(s, str) and s != {}:
            return s
        paths = self.load_path()
        return paths

    def load(self, suffix_or_path_or_state=None):
        s = self.load_state(suffix_or_path_or_state)
        log.info(f"Loading {s}.")
        s = torch.load(s, map_location='cuda:0')

    def save(self, info, suffix=None):
        os.makedirs(f'{self.model_dir}/', exist_ok=True)
        path = self.save_path(suffix)
        log.debug(f'Saving model {path!r}...')
        self.model.save(path, info)

    @property
    def len_epoch(self):
        return len(self.train_dataset.dataset)

    def epochs(self):
        return self.imgs_seen / self.len_epoch

    def eval_step(self, inputs, targets):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            probs = torch.nn.functional.log_softmax(outputs, dim=1)
            loss = torch.nn.functional.nll_loss(probs, targets)
            top1, top5  = topk(probs, targets, k=(1, 5))
            self.imgs_seen += targets.size(0)
        return (loss, top1, top5)

    def eval(self, suffix=None, key='eval', report_num_params=False):
        losses, accs, top1s, top5s = [], [], [], []
        for inputs, targets in self.test_dataset:
            loss, top1, top5 = self.eval_step(inputs, targets)
            losses.append(loss)
            top1s.append(top1)
            top5s.append(top5)
        avg_loss = torch.mean(torch.stack(losses))
        avg_top1, avg_top5 = np.mean(top1s)*100, np.mean(top5s)*100
        # log down results to tensorboard
        self.tb_writer.add_scalars(
            int(self.epochs()),
            {"eval_top1": avg_top1}, prefix="eval")
        self.tb_writer.add_scalars(
            int(self.epochs()),
            {"eval_top5": avg_top5}, prefix="eval")
        self.tb_writer.add_scalars(
            int(self.epochs()),
            {"eval_loss": avg_loss}, prefix="eval")
        log.info(f"Averaged accuracy top1/top5: {avg_top1:.2f}/{avg_top5:.2f}.")
        if report_num_params:
            num_params, flops = self.model.get_num_params_flops(inputs)
            log.info(f"Number of params {num_params}, number of Flops {flops}.")

    def info(self, summary_depth=None):
        hook = SummaryHook(summary_depth)
        self.eval(None, hook=hook)
        return hook.format()
