import torch
import numpy as np

from .table import Table


class SummaryHook:
    once = True

    def __init__(self, summary_depth=None):
        super().__init__()
        self.summary_depth = summary_depth or 1
        self.summary = {}
        self.info = {}
        self.depths = {}
        self.hooks = []

    def _register_hook(self, module, depth):
        # avoid = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
        # check = not isinstance(module, avoid)
        # check = check and (depth is None or depth >= 0)
        # check = check and module != self.model
        # check = check and getattr(module, 'enable_summary', True)
        self.depths[module] = depth
        module.register_forward_hook(self._hook)
        depth = depth if depth is None else depth - 1
        for m in module.children():
            self._register_hook(m, depth)

    def flatten(self, module):
        for m in module.children():
            self.flatten(m)
        if self.depths[module] > 0:
            return
        for m in module.children():
            child_info = self.info.pop(m, {})
            for k, v in child_info.items():
                info = self.info.setdefault(module, {})
                if not k.startswith('#') or k not in info:
                    info[k] = v
                else:
                    info[k] += v

    def hook(self, model):
        self.model = model
        self._register_hook(model, self.summary_depth)

    def unhook(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.flatten(self.model)

    def _hook(self, module, inputs, outputs):
        info = self.info.setdefault(module, {})
        info['input_shape'] = list(inputs[0].size())
        if isinstance(outputs, (list, tuple)):
            outshape = [list(o.size()) for o in outputs if o is not None]
        else:
            outshape = list(outputs.size())
        info['output_shape'] = outshape
        params = []
        for p in dir(module):
            p = getattr(module, p)
            if isinstance(p, torch.nn.Parameter):
                params.append(p)
            if isinstance(p, torch.nn.ParameterList):
                params += list(p)
        info['#params'] = 0
        info['#trainables'] = 0
        for p in params:
            num = int(np.product(p.size()))
            info['#params'] += num
            if p.requires_grad:
                info['#trainables'] += num
        if hasattr(module, 'estimate'):
            info.update(module.estimate(inputs, outputs))

    def format(self):
        table = Table(['name', 'shape', '#params', '#macs'])
        table.add_rule()
        table.add_row(
            ['input', self.info[self.model]['input_shape'], None, None])
        activations = 0
        for name, module in self.model.named_modules():
            if module == self.model or self.depths[module] < 0:
                continue
            try:
                info = self.info[module]
            except KeyError:
                continue
            row = [
                name, info.get('output_shape'),
                info.get('#params'), info.get('#macs')]
            table.add_row(row)
            activations += np.prod(info['output_shape'][1:])
        table.add_row(
            ['output', self.info[self.model]['output_shape'], None, None])
        table.footer_sum('#params')
        table.footer_sum('#macs')
        table.footer_value('shape', f'{int(activations):,}')
        return table.format()
