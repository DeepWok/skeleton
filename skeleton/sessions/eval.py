import torch
import os
import copy
import yaml
import numpy as np
from collections import defaultdict

from ..pretty import log
from .base import SessionBase


class Eval(SessionBase):
    def __init__(
            self, model_name, dataset_name,
            load_name=None, save_name=None,
            gpus=1, workers=None, batch_size=128,
            model_dir="models"):

        super().__init__(
            model_name, dataset_name,
            load_name, save_name, 
            gpus, workers, batch_size,
            max_epochs=None, model_dir=model_dir)
