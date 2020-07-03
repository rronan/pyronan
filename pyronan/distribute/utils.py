import argparse
import itertools
import os
from collections import OrderedDict
from copy import copy
from pathlib import Path
from pydoc import locate

import yaml

from pyronan.utils.misc import append_timestamp

parser_distribute = argparse.ArgumentParser()
parser_distribute.add_argument("config_path", type=Path, default="sweep.yaml")
parser_distribute.add_argument(
    "--log_dir", type=Path, default=os.environ.get("PYRONAN_LOG_DIR")
)
parser_distribute.add_argument("--ncpus", type=int, default=4)
parser_distribute.add_argument("--ngpus", type=int, default=1)
parser_distribute.add_argument("--jobs", type=int, default=1)
parser_distribute.add_argument("--merge_names", action="store_true")


def make_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "name" not in config:
        config["name"] = config_path.stem
    config["name"] = append_timestamp(config["name"], end=True)
    return config


def update_opt(opt, type_dict, dict_):
    for k, v in dict_.items():
        if type_dict[k] is not None:
            if type(v) is list:
                v = [type_dict[k](e) for e in v]
            else:
                v = type_dict[k](v)
        setattr(opt, k, v)
    return opt


def make_opt_list(config, merge_names):
    res = []
    opt, type_dict = locate(config["parser"])([])
    baseopt = update_opt(opt, type_dict, config["args"])
    for grid in config["grids"]:
        grid = OrderedDict(grid)
        for values in itertools.product(*grid.values()):
            opt = update_opt(copy(baseopt), type_dict, dict(zip(grid.keys(), values)))
            if merge_names:
                opt.name = config["name"]
            else:
                opt.name = "_".join([config["name"], f"{len(res):02d}"])
            res.append(opt)
    return res
