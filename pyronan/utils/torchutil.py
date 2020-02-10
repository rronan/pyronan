import json
import random
import time

import h5py
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from pyronan.utils.misc import args2dict, checkpoint


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
    def __init__(self, model=None, args=None):
        self.model = model
        self.args = args
        self.chkpt_interval = getattr(args, "subcheck")
        self.image_interval = getattr(args, "image_interval")
        self.step = 0
        self.tensorboard = None
        self.is_graph_written = False
        self.log = []
        if getattr(args, "tensorboard", False):
            self.tensorboard = SummaryWriter(log_dir=args.checkpoint)
            self.tensorboard.add_text(
                "args", json.dumps(args2dict(args), sort_keys=True, indent=4)
            )

    def start_epoch(self, i):
        self.t0 = time.time()
        self.log.append({"epoch": i})

    def end_epoch(self, i):
        self.log[i]["lr"] = self.model.get_lr()
        self.log[i]["time"] = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - self.t0)
        )
        checkpoint(f"{i:04d}", self.log, model=self.model, args=self.args)
        return self.log[i]

    def batch(self, loss, set_, i, j):
        for key, value in loss.items():
            self.log[i][f"{set_}_{key}"] = value
        if self.tensorboard is not None:
            for key, value in loss.items():
                self.tensorboard.add_scalar(f"{set_}_{key}", value, self.step)
        if (
            self.chkpt_interval is not None
            and j % self.chkpt_interval == 0
            and set_ == "train"
        ):
            self.checkpoint(i, self.log, f"{set_}_{j}")
        if (
            self.image_interval is not None
            and j % self.image_interval[set_] == 0
            and self.tensorboard is not None
        ):
            if hasattr(self.model, "get_image"):
                self.tensorboard.add_images(f"im_{set_}", self.model.get_image(0.3), i)
            self.add_graph()
            self.tensorboard.flush()

    def add_graph(self, overwrite=False):
        if self.is_graph_written and not overwrite:
            return
        try:
            if hasattr(self.model, "get_input_to_model"):
                self.tensorboard.add_graph(self.model, self.model.get_input_to_model())
                self.is_graph_written = True
        except Exception as e:
            print("Exception in Callback.add_graph():", e)
            self.is_graph_written = True


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
