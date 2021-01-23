import sys
import random
import argparse
import functools

import torch
import numpy as np

from .sessions import Train, Eval
from .pretty import log, csvread, plot


class Main:
    arguments = {
        ('action', ): {'type': str, 'help': 'Name of the action to perform.'},
        ('dataset', ): {'type': str, 'help': 'Name of the dataset.'},
        ('model', ): {'type': str, 'help': 'Name of the model.'},
        ('-load', '--load-name'): {
            'type': str, 'default': None,
            'help': 'Name of the saved model to restore.'
        },
        ('-save', '--save-name'): {
            'type': str, 'default': None,
            'help': 'Name of the saved model to save.'
        },
        ('-m', '--max-epochs'): {
            'type': float, 'default': 1000,
            'help': 'Maximum number of epochs for training.',
        },
        ('-b', '--batch-size'): {
            'type': int, 'default': 128,
            'help': 'Batch size for training and evaluation.',
        },
        ('-gpus', '--gpus'): {
            'type': int, 'default': 1,
            'help': 'Number of GPUs to use.',
        },
        ('-re', '--reset-epochs'): {
            'action': 'store_false',
            'help': 'Reset the number of epochs.',
        },
        ('-rb', '--reset-best'): {
            'action': 'store_true',
            'help':
                'Restart training with the current best model, '
                'and overwrite it.'
        },
        ('-d', '--debug'): {
            'action': 'store_true',
            'help': 'Verbose debug',
        },
        ('--summary-depth', ): {
            'type': int, 'default': None,
            'help': 'The depth for recursive summary.',
        },
        ('--deterministic', ): {
            'action': 'store_true',
            'help': 'Deterministic run for reproducibility.'
        },
        ('-lr', '--learning-rate'): {
            'type': float, 'default': 0.01, 'help': 'Initial learning rate.',
        },
        ('-lr_schedule', '--learning-rate-scheduler'): {
            'type': str, 'default': 'step', 'help': 'lr scheduler style.',
        },
        ('-lrd', '--learning-rate-decay-epochs'): {
            'type': int, 'default': 1000,
            'help': 'Number of epochs for each learning rate decay.',
        },
        ('-lrf', '--learning-rate-decay-factor'): {
            'type': float, 'default': 0.1,
            'help': 'Learning rate decay factor.',
        },
        ('-lreta', '--learning-rate-eta-min'): {
            'type': float, 'default': 0.0,
            'help': 'eta in cosine scheduler.',
        },
        ('-lrgamma', '--learning-rate-gamma'): {
            'type': float, 'default': 0.0,
            'help': 'eta in cosine scheduler.',
        },
        ('-w', '--workers'): {
            # multiprocessing fail with too many works
            'type': int, 'default': 10,
            'help': 'Number of CPU workers.',
        },
        ('-seed', '--seed'): {
            # random seed
            'type': int, 'default': 0,
            'help': 'Random seed.',
        },
        ('--model-dir', ): {
            'type': str, 'default': "models/",
            'help': 'Specifies the folder where to load and store models.'
        },
        ('--nolog', ):{
            'action':'store_true',
            'help': 'Turns off logging',
        }
    }

    def __init__(self):
        super().__init__()
        a = self.parse()

        if a.debug:
            log.level = 'debug'
            sys.excepthook = self._excepthook
        if a.nolog:
            log.level = 'off'
        if a.deterministic:
            seed = int(a.seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.a = a

    def parse(self):
        p = argparse.ArgumentParser(description='Low Precision Bayesian.')
        for k, v in self.arguments.items():
            p.add_argument(*k, **v)
        return p.parse_args()

    def _excepthook(self, etype, evalue, etb):
        from IPython.core import ultratb
        ultratb.FormattedTB()(etype, evalue, etb)
        for exc in [KeyboardInterrupt, FileNotFoundError]:
            if issubclass(etype, exc):
                sys.exit(-1)
        import ipdb
        ipdb.post_mortem(etb)

    def run(self):
        try:
            action = getattr(self, f'cli_{self.a.action.replace("-", "_")}')
        except AttributeError:
            callables = [n[4:] for n in dir(self) if n.startswith('cli_')]
            log.error_exit(
                f'Unkown action {self.a.action!r}, '
                f'accepts: {", ".join(callables)}.')
        return action()

    def train_session(self, session_cls, a, search_mode=True):
        if not a.save_name:
            log.error_exit('--save-name not specified.')

        lr_scheduler_map = {
            'step': functools.partial(
                torch.optim.lr_scheduler.StepLR,
                step_size=a.learning_rate_decay_epochs,
                gamma=a.learning_rate_decay_factor),
            'cosine': functools.partial(
                torch.optim.lr_scheduler.CosineAnnealingLR,
                T_max=a.max_epochs,
                eta_min=a.learning_rate_eta_min),
            'exponential': functools.partial(
                torch.optim.lr_scheduler.ExponentialLR,
                gamma=a.learning_rate_gamma),
            'cifar-multi-step': functools.partial(
                torch.optim.lr_scheduler.MultiStepLR,
                milestones=[100, 150], gamma=0.1),
            'cifar-search-multi-step': functools.partial(
                torch.optim.lr_scheduler.MultiStepLR,
                milestones=[100, 150], gamma=0.1),
        }
        lr_scheduler = lr_scheduler_map[a.learning_rate_scheduler]
        session = session_cls(
            a.model, a.dataset,
            a.load_name, a.save_name,
            a.gpus, a.workers, a.batch_size,
            a.learning_rate, lr_scheduler,
            max_epochs=a.max_epochs,
            debug=a.debug,
            model_dir=a.model_dir)
        if a.reset_epochs:
            session.reset_epochs()
        if a.reset_best:
            session.reset_best()
        return session

    def eval_session(self, a):
        return Eval(
            model_name=a.model,
            dataset_name=a.dataset,
            load_name=a.load_name,
            save_name=None,
            gpus=1,
            workers=a.workers,
            batch_size=a.batch_size,
            model_dir=a.model_dir)

    def cli_train(self):
        train = self.train_session(Train, self.a, search_mode=False)
        return train.train(self.a.max_epochs)

    def cli_test(self):
        return self.eval_session(self.a).run_eval(report_num_params=False)
    cli_eval = cli_test

    def cli_info(self):
        val = self.eval_session(self.a)
        return print(val.info(self.a.summary_depth))
