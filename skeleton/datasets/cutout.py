import torch
import numpy as np


class Cutout:
    """
    Randomly mask out one or more patches from an image.
    holes: Number of patches to cut out of each image.
    length: The length (in pixels) of each square patch.

    ref: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    """
    def __init__(self, holes=1, length=16):
        self.holes = holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask
