from argparse import ArgumentParser
from pathlib import Path
from pydoc import locate

import torch
import torch.optim as optim
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

from apex import amp
from pyronan.utils.misc import Nop

parser_model = ArgumentParser(add_help=False)
parser_model.add_argument("--grad_clip", type=float, default=None)
parser_model.add_argument(
    "--lr", nargs="+", type=lambda x: x.split(":"), default=[1e-3]
)
parser_model.add_argument("--lr_decay", type=float, default=0.1)
parser_model.add_argument("--lr_patience", type=int, default=10)
parser_model.add_argument("--optimizer", default="Adam")
parser_model.add_argument("--weight_decay", type=float, default=0)
parser_model.add_argument("--load", type=Path, default=None)
parser_model.add_argument("--data_parallel", action="store_true")
parser_model.add_argument("--gpu", action="store_true", help="Use NVIDIA GPU")
parser_model.add_argument("--amp_level", choices=["O0", "O1", "O2", "O3"], default=None)


def make_model(Model, args, gpu=False, data_parallel=False, load=None, amp_level=None):
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
        if nn_module is None:
            self.nn_module = nn.Module()
            self.optimizer = None
            self.lr_scheduler = Nop()
        else:
            self.set_optim(args)

    @staticmethod
    def _lr_arg(nn_module, kv):
        if type(kv) is float:
            return {"params": nn_module.parameters(), "lr": kv}
        if len(kv) == 1:
            return {"params": nn_module.parameters(), "lr": float(kv[0])}
        return {"params": getattr(nn_module, kv[0]).parameters(), "lr": float(kv[1])}

    @staticmethod
    def _lr_arg_list(nn_module, lr):
        if type(float) is float:
            return [{"params": nn_module.parameters(), "lr": lr}]
        return [Model._lr_arg(nn_module, kv) for kv in lr]

    @staticmethod
    def _check_parameter_lr(nn_module, lr):
        if type(lr) is float:
            return
        assert type(lr) is list
        num_param_total = sum([m.numel() for m in nn_module.parameters()])
        submodules = [k for k, _ in lr]
        num_param_list = [
            sum([m.numel() for m in getattr(nn_module, x).parameters()])
            for x in submodules
        ]
        assert num_param_total == sum(num_param_list)

    def set_optim(self, args):
        self.grad_clip = args.grad_clip
        self._check_parameter_lr(self.nn_module, args.lr)
        kwargs = {}
        if args.weight_decay is not None:
            kwargs["weight_decay"] = args.weight_decay
        lr = [self._lr_arg(self.nn_module, kv) for kv in args.lr]
        self.optimizer = getattr(optim, args.optimizer)(lr, **kwargs)
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=args.lr_patience,
            factor=args.lr_decay,
            verbose=True,
            eps=1e-9,
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

    def load(self, path):
        print(f"loading {path}")
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        state_dict = {k.replace(".module", "", 1): v for k, v in state_dict.items()}
        self.nn_module.load_state_dict(state_dict)

    def save(self, path, epoch):
        with open(path / f"{self.__class__.__name__}.txt", "w") as f:
            f.write(str(self))
        _state_dict = self.nn_module.state_dict()
        state_dict = {k.replace(".module", "", 1): v for k, v in _state_dict.items()}
        torch.save(state_dict, path / f"weights_{epoch}.pth")

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
            return 1

    def get_num_parameters(self):
        return sum([m.numel() for m in self.nn_module.parameters()])

    def __call__(self, *args, **kwargs):
        return self.nn_module.forward(*args, **kwargs)

    def requires_grad_(self, v):
        if self.is_data_parallel:
            self.nn_module.module.requires_grad_(v)
        else:
            self.nn_module.requires_grad_(v)
