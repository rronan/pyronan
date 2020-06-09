import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from pydoc import locate

import torch
import torch.optim as optim
from torch import nn
from torch.nn.utils import clip_grad_norm_

# from apex import amp
amp = None

parser_model = ArgumentParser(add_help=False)
parser_model.add_argument("--grad_clip", type=float, default=None)
parser_model.add_argument("--lr", type=json.loads, default=1e-3)
parser_model.add_argument("--optimizer", default="Adam")
parser_model.add_argument("--lr_scheduler", default="ReduceLROnPlateau")
parser_model.add_argument(
    "--lr_scheduler_kwargs",
    type=json.loads,
    default={"factor": 0.1, "patience": 10, "eps": 1e-9},
)
parser_model.add_argument("--weight_decay", type=float, default=0)
parser_model.add_argument("--load", type=Path, default=None)
parser_model.add_argument("--restore_optimizer", action="store_true")
parser_model.add_argument("--data_parallel", action="store_true")
parser_model.add_argument("--gpu", action="store_true", help="Use NVIDIA GPU")
parser_model.add_argument("--amp_level", choices=["O0", "O1", "O2", "O3"], default=None)


def make_model(
    Model,
    args,
    gpu=False,
    data_parallel=False,
    load=None,
    restore_optimizer=False,
    amp_level=None,
):
    if type(Model) is str:
        print("importing", Model)
        Model = locate(Model)
    model = Model(args)
    if load is not None:
        model.load(load)
    if data_parallel:
        model.data_parallel()
        print("Training on", torch.cuda.device_count(), "GPUs!")
    if gpu:
        model.gpu()
    if restore_optimizer:
        model.restore_optimizer(load)
    if amp_level is not None:
        model.amp(amp_level)
    print(f"n parameters: {model.get_num_parameters()}")
    return model


class Model:
    def __init__(self, nn_module=None, args=parser_model.parse_args([])):
        super().__init__()
        self.device = "cpu"
        self.is_data_parallel = False
        self.is_amp = False
        self.nn_module = nn_module
        if nn_module is not None:
            self.set_optim(args)

    @staticmethod
    def _lr_arg(nn_module, lr):
        if type(lr) is float or type(lr) is int:
            return [{"params": nn_module.parameters(), "lr": lr}]
        res = []
        for k, v in sorted(lr.items()):
            res.append({"params": getattr(nn_module, k).parameters(), "lr": v})
        return res

    def set_optim(self, args):
        self.grad_clip = args.grad_clip
        lr_arg = self._lr_arg(self.nn_module, args.lr)
        logging.info(lr_arg)
        wd = getattr(args, "weight_decay", 0)
        self.optimizer = getattr(optim, args.optimizer)(lr_arg, weight_decay=wd)
        if len(args.lr_scheduler) > 0:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(
                self.optimizer, **args.lr_scheduler_kwargs, verbose=True
            )

    def update(self, loss):
        self.nn_module.zero_grad()
        if self.is_amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if self.grad_clip is not None:
            clip_grad_norm_(self.nn_module.parameters(), self.grad_clip)
        self.optimizer.step()

    def step(self, batch, set_):
        self.x, self.y = batch[0].to(self.device), batch[1].to(self.device)
        if set_ == "train":
            self.pred = self.nn_module(self.x)
            loss = self.loss.forward(self.pred, self.y)
            self.update(loss)
        else:
            with torch.no_grad():
                self.pred = self.nn_module(self.x)
                loss = self.loss.forward(self.pred, self.y)
        return {"loss": loss.data.item()}

    def load(self, path, restore_optimizer=False):
        logging.info(f"loading {path}")
        chkpt = torch.load(path, map_location=self.device)
        if "weights" in chkpt:
            weights = {
                k.replace("module.", "", 1): v for k, v in chkpt["weights"].items()
            }
            self.nn_module.load_state_dict(weights)
        else:
            self.nn_module.load_state_dict(chkpt)

    def restore_optimizer(self, path):
        logging.info(f"restoring optimizer {path}")
        chkpt = torch.load(path, map_location=self.device)
        self.optimizer.load_state_dict(chkpt["optimizer"])

    def save(self, path, epoch):
        with open(path / f"{self.__class__.__name__}.txt", "w") as f:
            f.write(str(self))
        _state_dict = self.nn_module.state_dict()
        weights = {k.replace(".module", "", 1): v for k, v in _state_dict.items()}
        chkpt = {"weights": weights, "optimizer": self.optimizer.state_dict()}
        torch.save(chkpt, path / f"chkpt_{epoch}.pth")

    def to_device(self, batch):
        if type(batch) is list:
            return [b.to(self.device) for b in batch]
        if type(batch) is dict:
            return {k: v.to(self.device) for k, v in batch.items()}

    def cpu(self):
        self.nn_module.cpu()
        self.nn_module.device = "cpu"
        self.device = "cpu"

    def gpu(self):
        self.nn_module.cuda()
        self.nn_module.device = "cuda"
        self.device = "cuda"

    def data_parallel(self):
        self.nn_module = nn.DataParallel(self.nn_module)
        self.is_data_parallel = True

    def amp(self, amp_level):
        self.nn_module, self.optimizer = amp.initialize(
            self.nn_module, self.optimizer, opt_level=amp_level
        )
        self.is_amp = True

    def get_lr(self):
        if self.optimizer is not None:
            return self.optimizer.param_groups[0]["lr"]
        else:
            return 0

    def get_num_parameters(self):
        return sum([m.numel() for m in self.nn_module.parameters()])

    def __call__(self, *args, **kwargs):
        return self.nn_module.forward(*args, **kwargs)

    def requires_grad_(self, v):
        if self.is_data_parallel:
            self.nn_module.module.requires_grad_(v)
        else:
            self.nn_module.requires_grad_(v)
