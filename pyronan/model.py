from collections import OrderedDict
import torch
import torch.optim as optim
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchviz import make_dot, make_dot_from_trace


class dummy_scheduler:
    def step(*args, **kwargs):
        return None


class default_args_optim:
    optimizer = "Adam"
    lr = 0.001
    lr_decay = 0.1
    lr_patience = 10
    weight_decay = 0
    grad_clip = None


class Model(object):
    def __init__(self, core_module=None, args_optim=None):
        super().__init__()
        self.device = "cpu"
        self.is_data_parallel = False
        self.core_module = core_module
        if core_module is None:
            self.optimizer = None
            self.scheduler = dummy_scheduler
        else:
            self.set_optim(args_optim)

    def set_optim(self, args_optim=None):
        if args_optim is None:
            print("args_optim set to default", default_args_optim)
            args_optim = default_args_optim
        self.grad_clip = args_optim.grad_clip
        kwargs = {}
        if args_optim.lr is not None:
            kwargs["lr"] = args_optim.lr
        if args_optim.weight_decay is not None:
            kwargs["weight_decay"] = args_optim.weight_decay
        self.optimizer = getattr(optim, args_optim.optimizer)(
            self.core_module.parameters(), **kwargs
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=args_optim.lr_patience,
            factor=args_optim.lr_decay,
            verbose=True,
            eps=1e-9,
        )

    def update(self, loss):
        self.core_module.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            clip_grad_norm_(self.core_module.parameters(), self.grad_clip)
        self.optimizer.step()

    def step(self, batch, set_):
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        pred = self.core_module(x)
        loss = self.loss.forward(pred, y)
        if set_ == "train":
            self.update(loss)
        return {"loss": loss.data.item()}

    def load(self, path):
        print(f"loading {path}")
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        try:
            self.core_module.load_state_dict(state_dict)
        except RuntimeError as e:
            print(e)
            state_dict = OrderedDict(
                {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            )
            self.core_module.load_state_dict(state_dict)

    def save(self, path, epoch):
        with open(path / f"{self.__class__.__name__}.txt", "w") as f:
            f.write(str(self))
        if self.is_data_parallel:
            state_dict = self.core_module.module.state_dict()
        else:
            state_dict = self.core_module.state_dict()
        torch.save(state_dict, path / f"weights_{epoch}.pth")

    def to_device(self, batch):
        return [b.to(self.device) for b in batch]

    def gpu(self):
        self.core_module.cuda()
        self.core_module.device = "cuda"
        self.device = "cuda"

    def data_parallel(self):
        self.core_module = nn.DataParallel(self.core_module)
        self.is_data_parallel = True

    def get_lr(self):
        if self.optimizer is not None:
            return self.optimizer.param_groups[0]["lr"]
        else:
            return 0

    def get_num_parameters(self):
        return sum([m.numel() for m in self.core_module.parameters()])

    def __call__(self, *args, **kwargs):
        return self.core_module.forward(*args, **kwargs)
