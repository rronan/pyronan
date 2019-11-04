from argparse import ArgumentParser

import numpy as np
import torch
import torch.utils.data
from path import Path

from pyronan.utils.misc import parse_slice

parser = ArgumentParser(add_help=False)
parser.add_argument("--frame_prefix", type=Path, default=None)
parser.add_argument("--input_prefix", type=Path, default=None)
parser.add_argument("--item_list", nargs="+", default=[])
parser.add_argument("--hw", type=int, default=128)
parser.add_argument("--nc_in", type=int, default=None)
parser.add_argument("--nc_out", type=int, default=None)
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--slice", type=parse_slice, default=slice(None))
parser.add_argument("--clip", type=float, default=10)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, set_):
        self.set_ = set_
        self.item_list = args.item_list
        self.getfunc_list = [getattr(self, "get_" + item) for item in args.item_list]

    def make_data(self, set_, *args, **kwargs):
        self.data = None

    def count_data(self, *args, **kwargs):
        self.count = None

    def print_stats(self):
        d = {"input": self.data}
        print("*****", self.set_, sep="\n")
        for key, value in d.items():
            print(f"{key} max :", np.max(value))
            print(f"{key} min :", np.min(value))
            print(f"{key} mean :", np.mean(value))
            print(f"{key} std :", np.std(value))
            print(f"{key} shape :", value.shape)
        print(f"n samples {self.set_}: {self.count}")

    def get_input(self, index):
        raise NotImplementedError

    def get_target(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        item_list = [f(index) for f in self.getfunc_list]
        return tuple(item_list)

    def __len__(self):
        return self.count
