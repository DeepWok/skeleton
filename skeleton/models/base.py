import torch
from torch import nn
# from thop import profile
from ..utils import summarize


class ModelBase(nn.Module):
    name = None
    input_size = None

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def load_state(self, state):
        saved = {n: p.shape for n, p in state.items()}
        ours = {n: p.shape for n, p in self.state_dict().items()}
        missing = [n for n in ours if n not in saved]

        summarize('Missing parameters', missing, level='warn')
        unexpected = [n for n in saved if n not in ours]
        summarize('Unused parameters', unexpected, level='warn')
        diffshape = [
            n for n, s in ours.items()
            if n in saved and tuple(s) != tuple(saved[n])]
        summarize('Parameters with mismatched shapes', diffshape, level='warn')
        self.load_state_dict(state, strict=False)

    def save(self, path, info):
        state = self.state_dict()
        state['.info'] = info
        torch.save(state, path)
    
    def get_num_params_flops(self, inputs):
        pass
        # macs, params = profile(self, inputs=(inputs, ))
