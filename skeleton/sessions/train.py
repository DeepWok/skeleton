import os
import yaml
import torch
from torch import nn

from ..pretty import log, MovingAverage, History
from ..utils import topk, get_num_params
from .base import SessionBase


class Train(SessionBase):

    is_training = True
    len_history = 100
    weight_decay = 1e-5

    def __init__(
            self, model_name, dataset_name,
            load_name=None, save_name=None,
            gpus=1, workers=None, batch_size=128,
            lr=0.01, 
            lr_scheduler=None,
            max_epochs=None,
            debug=False,
            model_dir="models/"):
        super().__init__(
            model_name, dataset_name,
            load_name, save_name, 
            gpus, workers, batch_size,
            max_epochs,
            model_dir=model_dir)
        # self.train_dataset, self.test_dataset = self.Dataset()
        if os.path.exists(self.save_path()[0]):
            with log.use_pause_level('warn'):
                log.warn(
                    'This session will overwrite the existing saved model '
                    f'{self.save_path()!r}, are you sure?')
        self.best_epochs = self.best_acc = None
        self.cur_acc = -1
        self.imgs_seen = 0
        if self._info is not None:
            top1 = self._info.get('top1')
            top5 = self._info.get('top5')
            self.best_acc = top1
            self.best_epochs = self._info.get('epochs', 0)
            self.imgs_seen = self._info.get('imgs_seen', 0)
            text = f'Resuming saved model'
            if top1 and top5:
                text += f' with {top1:.2%} top-1 '
                text += f'and {top5:.2%} top-5 accuracies'
            log.info(f'{text}...')

        train_optimizer = torch.optim.SGD(
            self.model.parameters(), momentum=0.9,
            lr=lr,
            weight_decay=1e-4)
            # weight_decay=5e-5)

        self.optimizers = {
            "train": train_optimizer}

        self.lr_schedulers = {}
        if lr_scheduler is not None:
            self.lr_schedulers['train'] = lr_scheduler(self.optimizers['train'])

        self.averages = {
            'train': {
                'loss': MovingAverage(self.len_history),
                'acc': MovingAverage(self.len_history),
            },
        }
        self.histories = {
            'train': History(
                self.prefix, self.load_name, self.save_name, 'train',
                ['epochs', 'train_loss', 'train_acc']),
            'eval': History(
                self.prefix, self.load_name, self.save_name, 'eval',
                ['epochs', 'top1', 'top5'])
        }
        self.criterion = nn.CrossEntropyLoss()
        self.debug = debug

    def save(self, info, suffix=None):
        info['epochs'] = self.epochs()
        info['imgs_seen'] = self.imgs_seen
        super().save(info, suffix)

    def reset_epochs(self):
        log.info('Resetting the number of epochs...')
        self.imgs_seen = 0
        for lr in self.lr_schedulers.values():
            lr.last_epoch = 0

    def reset_best(self):
        log.info('Resetting the best model...')
        self.best_epochs = self.best_acc = None

    def save_epoch(self, suffix):
        top1, top5, loss = self.eval(suffix)
        info = {
            'event': 'latest', 'epochs': self.epochs(),
            'top1': top1, 'top5': top5}
        self.histories['eval'].record(info)
        for h in self.histories.values():
            h.flush()
        self.save(info, f'{suffix}.last' if suffix else 'last')

        self.cur_acc = top1
        best_top1 = self.best_acc if self.best_acc is not None else 0
        if top1 <= best_top1:
            return
        log.debug(
            f'Saving the current best model at {self.epochs()} epochs '
            f'with {top1:.2%} accuracy.')
        info['event'] = 'best'
        self.save(info, suffix)
        self.best_acc = top1
        self.best_epochs = self.epochs()

    def save_last(self, suffix=None):
        if not log.countdown('Saving the latest model', 3):
            return
        info = {'event': 'max_epochs', 'top1': None, 'top5': None}
        self.save(info, f'{suffix}.last' if suffix else 'last')

    def loop(self, epoch, max_epochs=None, suffix=None):
        try:
            while True:
                epoch(suffix)
                if max_epochs is not None and self.epochs() > max_epochs:
                    log.info(f'Maximum number of epochs {max_epochs} reached.')
                    break
        except KeyboardInterrupt:
            log.debug(f'Abort.')

    def train(self, max_epochs=None, suffix=None):
        log.debug(f'Training with a dataset of length {self.len_epoch}...')
        for lr in self.lr_schedulers.values():
            lr.last_epoch = self.epochs()
        self.loop(self.train_epoch, max_epochs, suffix)
        self.save_last(suffix)
        log.info(f"Best test acc is {self.best_acc}")
        self.save_train_info()

    def save_train_info(self):
        info = {
            'parameters': get_num_params(self.model),
            'acc': self.best_acc,
        }
        with open("train_info.yaml", 'w') as f:
            yaml.dump(info, f)

    def selector_loss(self):
        loss = 0
        for l in self.model.layers:
            loss += l.mixture.selector_loss
        return loss

    def lists_to_dict(self, names, values):
        tmp = {}
        for n, v in zip(names, values):
            tmp[n] = v
        return tmp

    def _step(self, key, data, target, retain_graph=False):
        for _key in self.optimizers:
            self.optimizers[_key].zero_grad()

        outputs = self.model(data)
        loss = self.criterion(outputs, target)

        if torch.isnan(loss):
            raise ValueError('Training loss diverged to NaN.')
        acc = topk(outputs, target)[0]
        loss.backward()

        self.optimizers[key].step()
        self.averages[key]['loss'].add(loss)
        self.averages[key]['acc'].add(acc)
        return loss, acc

    def train_epoch(self, suffix):
        self.model.train()
        for inputs, targets in self.train_dataset:
            loss, acc = self._step(
                'train', inputs, targets)
            info = {
                'epochs': self.epochs(),
                'train_loss': float(loss),
                'train_acc': float(acc),
            }
            self.histories['train'].record(info)
            text = [
                f'epochs: {self.epochs():.2f}',
                f'lr: {self.lr_schedulers["train"].get_lr()[-1]:.2e}',
                f'loss: {self.averages["train"]["loss"]:s}',
                f'train acc: {float(self.averages["train"]["acc"]):.2%}',
            ]

            if self.best_acc is not None and self.best_epochs is not None and self.cur_acc is not None:
                text.append(
                    f'val acc: {self.cur_acc:.2%} / best: {self.best_acc:.2%} [{int(self.best_epochs)}]')
            log.info(', '.join(text), update=True)
            self.imgs_seen += targets.size(0)
            if self.debug:
                break
        self.save_epoch(suffix)
        self.lr_schedulers['train'].step()
