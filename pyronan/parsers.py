from argparse import ArgumentParser
from path import Path
from time import strftime


from pyronan.utils.misc import append_timestamp


parser_optim = ArgumentParser(add_help=False)
parser_optim.add_argument("--grad_clip", type=float, default=None)
parser_optim.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser_optim.add_argument("--lr_decay", type=float, default=0.1)
parser_optim.add_argument("--lr_patience", type=int, default=10)
parser_optim.add_argument("--optimizer", default="Adam")
parser_optim.add_argument("--weight_decay", type=float, default=0)


parser_train = ArgumentParser(add_help=False)
parser_train.add_argument("--seed", type=int, default=0)
parser_train.add_argument("--bsz", type=int, help="batch size")
parser_train.add_argument("--checkpoint", type=Path)
parser_train.add_argument("--checkpoint_interval", type=int, default=1)
parser_train.add_argument("--data_parallel", action="store_true")
parser_train.add_argument("--gpu", action="store_true", help="Use NVIDIA GPU")
parser_train.add_argument(
    "--name", type=append_timestamp, default=strftime("%y%m%d_%H%M%S")
)
parser_train.add_argument("--num_workers", type=int, default=20)
parser_train.add_argument("--n_epochs", type=int, default=200)
parser_train.add_argument("--pin_memory", action="store_true")
parser_train.add_argument("--save_all", action="store_true")
parser_train.add_argument("--save_last", action="store_true")
parser_train.add_argument("--subcheck", type=float, default=None)
