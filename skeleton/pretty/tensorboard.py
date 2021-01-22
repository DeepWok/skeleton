import torch
import tensorboardX
import datetime
import numpy as np

from ..utils import device

class TensorboardWriter:
    def __init__(
            self, dataset_name, model_name,
            time_stamp=True):
        now = datetime.datetime.now()
        log_dir = f"tb_summary/{dataset_name.lower()}_{model_name}"
        # resolution down to minutes
        log_dir += "_%02d%02d"%(now.month, now.day)
        log_dir += "_%02d%02d"%(now.hour, now.minute)
        self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)

    def add_scalars(self, i, key_value_map, prefix=None):
        for k, v in key_value_map.items():
            if prefix is None:
                self.writer.add_scalar(f"scalars/{k}", v, i)
            else:
                self.writer.add_scalar(f"{prefix}/{k}", v, i)

    def add_op_distributions(
            self, i, priors, prefix=None,
            profile_freq=1, sample_size=1000):
        # profiling distribution might be expensive, do it less frequently
        if i % profile_freq == 0:
            # multi-nomial distribution sampled with sample_size points
            for j, prior in enumerate(priors):
                name, prior = f'layer_{j}', prior
                if torch.sum(prior) > 0:
                    forced_range = torch.tensor(np.arange(0, len(prior)), device=device)
                    multi = torch.multinomial(prior, sample_size, replacement=True)
                    multi = torch.cat([multi, forced_range])
                    self.writer.add_histogram(
                        f"{prefix}/{name}", multi, i)


