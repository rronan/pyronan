from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser_optim = ArgumentParser(add_help=False)
parser_optim.add_argument("--grad_clip", type=float, default=None)
parser_optim.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser_optim.add_argument("--lr_decay", type=float, default=0.1)
parser_optim.add_argument("--lr_patience", type=int, default=10)
parser_optim.add_argument("--optimizer", default="Adam")
parser_optim.add_argument("--weight_decay", type=float, default=0)


class dummy_scheduler:
    def step(*args, **kwargs):
        return None


class Model(object):
    def __init__(self, nn_module=None, args_optim=None):
        super().__init__()
        if args_optim is None:
            args_optim = parser_optim.parse_args()
        self.device = "cpu"
        self.is_data_parallel = False
        self.nn_module = nn_module
        if nn_module is None:
            self.optimizer = None
            self.scheduler = dummy_scheduler
        else:
            self.set_optim(args_optim)

    def set_optim(self, args_optim):
        self.grad_clip = args_optim.grad_clip
        kwargs = {}
        if args_optim.lr is not None:
            kwargs["lr"] = args_optim.lr
        if args_optim.weight_decay is not None:
            kwargs["weight_decay"] = args_optim.weight_decay
        self.optimizer = getattr(optim, args_optim.optimizer)(
            self.nn_module.parameters(), **kwargs
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=args_optim.lr_patience,
            factor=args_optim.lr_decay,
            verbose=True,
            eps=1e-9,
        )

    def update(self, loss):
        self.nn_module.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            clip_grad_norm_(self.nn_module.parameters(), self.grad_clip)
        self.optimizer.step()

    def step(self, batch, set_):
        self.x, self.y = batch[0].to(self.device), batch[1].to(self.device)
        self.pred = self.nn_module(self.x)
        loss = self.loss.forward(self.pred, self.y)
        if set_ == "train":
            self.update(loss)
        return {"loss": loss.data.item()}

    def load(self, path):
        print(f"loading {path}")
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.nn_module.load_state_dict(state_dict)

    def save(self, path, epoch):
        with open(path / f"{self.__class__.__name__}.txt", "w") as f:
            f.write(str(self))
        if self.is_data_parallel:
            state_dict = self.nn_module.module.state_dict()
        else:
            state_dict = self.nn_module.state_dict()
        torch.save(state_dict, path / f"weights_{epoch}.pth")

    def to_device(self, batch):
        return [b.to(self.device) for b in batch]

    def gpu(self):
        self.nn_module.cuda()
        self.nn_module.device = "cuda"
        self.device = "cuda"

    def data_parallel(self):
        self.nn_module = nn.DataParallel(self.nn_module)
        self.is_data_parallel = True

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
