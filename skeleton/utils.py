import types
import functools

import torch
from torch.nn.utils.rnn import pad_sequence


use_cuda = torch.cuda.is_available()
print('Using cuda:{}'.format(use_cuda))
torch_cuda = torch.cuda if use_cuda else torch
device = torch.device('cuda' if use_cuda else 'cpu')


def to_device(x):
    if use_cuda:
        x.to(device)


def to_numpy(x):
    if use_cuda:
        x = x.cpu()
    return x.detach().numpy()


def get_num_params(module_cls):
    pp=0
    for p in list(module_cls.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def summarize(msg, names, count=5, level='info'):
    from .pretty import log
    if not names:
        return
    count = min(len(names), count)
    diff = len(names) - count
    names = '\n    '.join(names[:count])
    text = f'{msg}:\n    {names}'
    if diff > 0:
        text += f' [{diff} more...]'
    getattr(log, level)(text)


def memoize(func):
    """
    A decorator to remember the result of the method call
    """
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        name = '_memoize_{}'.format(func.__name__)
        try:
            return getattr(self, name)
        except AttributeError:
            result = func(self, *args, **kwargs)
            if isinstance(result, types.GeneratorType):
                # automatically resolve generators
                result = list(result)
            setattr(self, name, result)
            return result
    return wrapped


def memoize_property(func):
    return property(memoize(func))


def topk(output, target, k=(1, ), count=False, graph_mask=None):
    try:
        _, pred = output.topk(max(k), 1, True, True)
    except:
        import pdb; pdb.set_trace()
    # if graph_mask is not None:
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    batch = 1 if count else target.size(0)
    return [float(correct[:k].sum()) / batch for i, k in enumerate(k)]


def onehot(labels, num_classes):
    output = torch_cuda.FloatTensor(labels.size(0), num_classes).zero_()
    return output.scatter_(1, labels.unsqueeze(-1), 1)


def safe_log(value, epsilon=1e-20):
    return torch.log(torch.clamp(value, min=epsilon))


def gumbel(shape, epsilon=1e-20):
    uniform = torch.rand(shape)
    if use_cuda:
        uniform = uniform.cuda()
    return -safe_log(-safe_log(uniform))


def fuzzy_softmax(probs_or_logits, temperature=1.0):
    shape = probs_or_logits.shape
    sumed = torch.nn.functional.softmax(probs_or_logits, dim=0) + temperature * torch.rand(shape).to(device)
    # renorm back to 0-1
    sumed = sumed / torch.sum(sumed)
    return sumed


def gumbel_softmax(probs_or_logits, temperature=1.0, log=True, epsilon=1e-20):
    logits = safe_log(probs_or_logits, epsilon) if log else probs_or_logits
    output = logits + gumbel(logits.shape)
    return torch.nn.functional.softmax(output / temperature, dim=-1)


def gumbel_max(probs_or_logits, temperature=1.0, log=True, epsilon=1e-20):
    logits = safe_log(probs_or_logits, epsilon) if log else probs_or_logits
    if temperature == 0:
        return torch.max(logits, dim=-1)
    if temperature != 1:
        logits /= temperature
    if temperature > 100:
        return torch.rand(logits.shape)
    return torch.max(logits + gumbel(logits.shape, 1e-20), dim=-1)


class DataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        obj = self.__dict__['_modules']['module']
        return obj if name == 'module' else getattr(obj, name)

