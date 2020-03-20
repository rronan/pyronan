import random

import h5py
import torch
import torch.nn as nn


def set_seed(seed, gpu):
    random.seed(seed)
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed_all(seed)


def obj_nan(x):
    x_flat = x.contiguous().view(x.size(0), x.size(1), -1)
    isnan = (x_flat != x_flat).any(-1)
    for _ in range(len(x.shape) - len(isnan.shape)):
        isnan = isnan.unsqueeze(-1)
    isnan = isnan.expand(x.shape)
    return isnan


def fillnan(tensor):
    return tensor.masked_fill(obj_nan(tensor), 0)


def load_from_keras(self, h5_path):
    print("loading weights from %s" % h5_path)
    f = h5py.File(h5_path)
    k = 1
    numel = 0
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            w = f["model_weights"]["conv2d_%d" % k]["conv2d_%d" % k]
            m.weight.data.copy_(
                torch.FloatTensor(w["kernel:0"].value).permute(3, 2, 0, 1)
            )
            m.bias.data.copy_(torch.FloatTensor(w["bias:0"].value))
            numel += m.weight.data.numel()
            numel += m.bias.data.numel()
            k += 1
    try:
        w = f["model_weights"]["conv2d_%d" % k]["conv2d_%d" % k]["kernel:0"]
        print("test failed: ", w.value)
    except Exception:
        print("success, number of parameters copied: %d" % numel)


def is_backbone_grad(lr):
    return type(lr) is float or type(lr) is int or "backbone" in lr
