import random

import h5py
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from pyronan.utils.misc import checkpoint


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


class Callback:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.interval = getattr(args, "subcheck")
        self.step = 0
        self.tensorboard = None
        if getattr(args, "tensorboard", False):
            self.tensorboard = SummaryWriter(log_dir=args.checkpoint)

    def add_scalar_dict(self, loss_dict, set_):
        for key, value in loss_dict.items():
            self.tensorboard.add_scalar(f"{set_}_{key}", value, self.step)

    def checkpoint(self, epoch, log, tag=""):
        checkpoint(f"{epoch:03d}_{tag}", log, model=self.model, args=self.args)
        if self.tensorboard is not None:
            if hasattr(self.model, "get_image"):
                self.tensorboard.add_images(f"im_{tag}", self.model.get_image(), epoch)
            self.tensorboard.flush()


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
